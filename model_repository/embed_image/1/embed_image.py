import asyncio
import base64
from io import BytesIO
import json
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for siglip model
    """

    def initialize(self, args):
        """_summary_

        Parameters
        ----------
        args : _type_
            _description_
        """
        self.model_config = model_config = json.loads(args["model_config"])
        embedding_config = pb_utils.get_output_config_by_name(model_config, "EMBEDDING")
        self.embedding_dtype = pb_utils.triton_string_to_numpy(
            embedding_config["data_type"]
        )
        model_path = model_config["parameters"]["model_path"]["string_value"]
        # bool_value doesn't appear supported forcing using string_value
        base64_encoded_default_str = model_config["parameters"][
            "base64_encoded_default"
        ]["string_value"]
        if base64_encoded_default_str.lower() == "true":
            self.base64_encoded_default = True
        elif base64_encoded_default_str.lower() == "false":
            self.base64_encoded_default = False
        else:
            raise pb_utils.TritonError(
                "model_config['parameters']['base64_encoded_default']="
                + f"{base64_encoded_default_str} must be 'true' | 'false'. "
            )

        self.default_embed_model = model_config["parameters"]["default_embed_model"][
            "string_value"
        ]
        self.processors = {}
        self.processors["siglip"] = AutoProcessor.from_pretrained(
            model_path,
            local_files=True,
        )

    def process_request(self, request, embed_model: str, base64_encoded: bool):
        try:
            input_image_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
        except Exception as exc:
            raise ValueError(f"Failed on getting input tensor from request: {exc}")

        try:
            images_bytes = []
            for b in input_image_tt.as_numpy().reshape(-1):
                if base64_encoded:
                    images_bytes.append(base64.b64decode(b))
                else:
                    images_bytes.append(b)
        except Exception as exc:
            raise ValueError(
                f"Failed getting bytes of the image from request data: {exc}"
            )

        try:
            images = [Image.open(BytesIO(b)).convert("RGB") for b in images_bytes]
        except Exception as exc:
            raise ValueError(f"Failed on PIL.Image.open() request data: {exc}")

        try:
            pixel_values_np = self.processors[embed_model](
                images=images, padding="max_length", return_tensors="pt"
            )["pixel_values"].numpy()
        except Exception as exc:
            raise ValueError(f"Failed on SiglipProcessor(images=image): {exc}")

        # Shape = [batch_size, 3, 384, 384], where batch_size should be 1
        return pixel_values_np

    async def execute(self, requests: list) -> list:
        """
        This is a BLS deployment that allows clients to send either base64 encoded or
        raw bytes of an image file. This does the processing of the image to get the
        inputs to be sent to the Siglip deployment, which allows for future conversion
        to onnx for the GPU workload. This supports dynamic batching of images for
        processing.

        Parameters
        ----------
        requests : list
            List of pb_utils.InferenceRequest containing an image to be embedded.

        Returns
        -------
        list
            List of pb_utils.InferenceResponse to be sent back to clients of the
            embedding vector for the given image sent in the InferenceRequest.
        """
        logger = pb_utils.Logger
        batch_size = len(requests)
        logger.log_info(f"embed_image.execute received {batch_size} requests")
        responses = [None] * batch_size
        inference_response_awaits = []
        valid_requests = []
        for batch_id, request in enumerate(requests):
            # Handle any request parameters
            request_params = json.loads(request.parameters())
            base64_encoded = request_params.get(
                "base64_encoded", self.base64_encoded_default
            )
            embed_model = request_params.get("embed_model", self.default_embed_model)

            try:
                pixel_values_np = self.process_request(
                    request, embed_model, base64_encoded
                )
                pixel_values_tt = pb_utils.Tensor("PIXEL_VALUES", pixel_values_np)
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"{exc}")
                )
                responses[batch_id] = response
            else:
                # Submit the request to siglip
                infer_siglip_request = pb_utils.InferenceRequest(
                    model_name=embed_model,
                    requested_output_names=["EMBEDDING"],
                    inputs=[pixel_values_tt],
                )
                # Perform asynchronous inference request
                inference_response_awaits.append(infer_siglip_request.async_exec())
                valid_requests.append(batch_id)

        inference_responses = await asyncio.gather(*inference_response_awaits)
        for siglip_response, batch_id in zip(inference_responses, valid_requests):
            if siglip_response.has_error() and responses[batch_id] is None:
                err_msg = (
                    "Error embedding the image with siglip: "
                    + f"{siglip_response.error().message()}"
                )
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(err_msg)
                )
                responses[batch_id] = response
            else:
                embedding_tt = pb_utils.get_output_tensor_by_name(
                    siglip_response, "EMBEDDING"
                )
                # embedding_np = (
                #    pb_utils.get_output_tensor_by_name(siglip_response, "EMBEDDING")
                #    .as_numpy()
                #    .reshape(-1)
                # )
                # embedding_tt = pb_utils.Tensor("EMBEDDING", embedding_np)
                response = pb_utils.InferenceResponse(output_tensors=[embedding_tt])
                responses[batch_id] = response

        return responses
