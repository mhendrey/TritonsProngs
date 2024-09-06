import asyncio
import base64
from io import BytesIO
import json
from PIL import Image
from transformers import AutoProcessor

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for image embedding
    models. Currently only SigLIP is supported.
    """

    def initialize(self, args):
        """
        Initialize embedding models' processors and load configuration parameters.

        Parameters
        ----------
        args : dict
            Command-line arguments for launching Triton Inference Server
        """
        self.model_config = model_config = json.loads(args["model_config"])
        embedding_config = pb_utils.get_output_config_by_name(model_config, "EMBEDDING")
        self.embedding_dtype = pb_utils.triton_string_to_numpy(
            embedding_config["data_type"]
        )

        ## Load the different models needed for processing inputs
        # Currently just the one model, but this is how to add additional ones
        self.processors = {}
        self.processors["siglip_vision"] = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", local_files=True)
                
        ## Get additional parameters from the config.pbtxt file
        # bool_value doesn't appear supported forcing using string_value
        # Specify the default value for base64_encoded request parameter.
        default_base64_encoded_str = model_config["parameters"][
            "default_base64_encoded"
        ]["string_value"]
        if default_base64_encoded_str.lower() == "true":
            self.default_base64_encoded = True
        elif default_base64_encoded_str.lower() == "false":
            self.default_base64_encoded = False
        else:
            raise pb_utils.TritonError(
                "model_config['parameters']['default_base64_encoded']="
                + f"{default_base64_encoded_str} must be 'true' | 'false'. "
            )

        # Specify the default embedding model. Can be overriden in request parameter
        self.default_embed_model = model_config["parameters"]["default_embed_model"][
            "string_value"
        ]

    def process_request(self, request, embed_model: str, base64_encoded: bool):
        """
        Process the input image request and prepare pixel values for embedding.

        Parameters
        ----------
        request : pb_utils.InferenceRequest
            Inference request containing the input image.
        embed_model : str
            Embedding model to use.
        base64_encoded : bool
            Whether the input image is base64 encoded.

        Returns
        -------
        np.ndarray
            Processed pixel values of the input image.
        """
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
            if embed_model == "siglip_vision":
                pixel_values_np = self.processors[embed_model](
                    images=images, padding="max_length", return_tensors="pt"
                )["pixel_values"].numpy()
        except Exception as exc:
            raise ValueError(
                f"Failed on {embed_model}'s Processor(images=image): {exc}"
            )

        # Shape = [batch_size, 3, 384, 384], where batch_size should be 1
        return pixel_values_np

    async def execute(self, requests: list) -> list:
        """
        Execute a batch of embedding requests on provided images. Images can be
        sent either as raw bytes or base64 encoded strings.

        Option Request Parameters
        -------------------------
        embed_model : str
            Specify which embedding model to use.
            If None, default_embed_model is used.
        base64_encoded : bool
            Set to true if image is sent base64 encoded.
            If None, default_base64_encoded is used.


        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]
            List of inference requests each containing an image to be embedded.

        Returns
        -------
        List[pb_utils.InferenceResponse]
            List of response objects with embedding results or error messages
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
                "base64_encoded", self.default_base64_encoded
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
                # Submit the request to embed_model
                infer_model_request = pb_utils.InferenceRequest(
                    model_name=embed_model,
                    requested_output_names=["EMBEDDING"],
                    inputs=[pixel_values_tt],
                )
                # Perform asynchronous inference request
                inference_response_awaits.append(infer_model_request.async_exec())
                valid_requests.append(batch_id)

        inference_responses = await asyncio.gather(*inference_response_awaits)
        for model_response, batch_id in zip(inference_responses, valid_requests):
            if model_response.has_error() and responses[batch_id] is None:
                err_msg = (
                    "Error embedding the image: "
                    + f"{model_response.error().message()}"
                )
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(err_msg)
                )
                responses[batch_id] = response
            else:
                embedding_tt = pb_utils.get_output_tensor_by_name(
                    model_response, "EMBEDDING"
                )
                response = pb_utils.InferenceResponse(output_tensors=[embedding_tt])
                responses[batch_id] = response

        return responses
