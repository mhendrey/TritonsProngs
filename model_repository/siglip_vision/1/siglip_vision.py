import json
import torch
from transformers import SiglipVisionModel

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for siglip vision
    model.
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

        # Use the GPU if available, otherwise use the CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.torch_dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            self.torch_dtype = torch.float32  # CPUs can't handle float16

        self.model = SiglipVisionModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=self.torch_dtype,
            local_files_only=True,
            use_safetensors=True,
        )
        # If on a GPU, use torch.compile to improve throughput
        if torch.cuda.is_available():
            self.model = torch.compile(self.model, dynamic=True)

    def execute(self, requests: list) -> list:
        """_summary_

        Parameters
        ----------
        requests : list
            _description_

        Returns
        -------
        list
            _description_
        """
        logger = pb_utils.Logger
        batch_size = len(requests)
        logger.log_info(f"siglip_vision.execute received {batch_size} requests")
        responses = [None] * batch_size
        batch_pixel_values = []
        valid_requests = []
        for batch_id, request in enumerate(requests):
            try:
                pixel_values_pt = torch.from_numpy(
                    pb_utils.get_input_tensor_by_name(
                        request, "PIXEL_VALUES"
                    ).as_numpy()
                )
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"{exc}")
                )
                responses[batch_id] = response
            else:
                batch_pixel_values.append(pixel_values_pt)
                valid_requests.append(batch_id)

        # Create batch to be processed shape=[len(valid_requests), 3, 384, 384]
        batch_pixel_values = (
            torch.cat(batch_pixel_values, dim=0).type(self.torch_dtype).to(self.device)
        )
        try:
            with torch.no_grad():
                images_embedding_np = (
                    self.model(pixel_values=batch_pixel_values)["pooler_output"]
                    .cpu()
                    .type(torch.float32)
                    .numpy()
                )
        except Exception as exc:
            # Problem embedding the whole batch. They all failed
            for i in valid_requests:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        "Siglip_vision threw error embedding the batch. Check your "
                        + "input image and/or try again"
                    )
                )
                responses[i] = response
            return responses

        for i, embedding in zip(valid_requests, images_embedding_np):
            embedding_tt = pb_utils.Tensor("EMBEDDING", embedding.reshape(1, -1))
            response = pb_utils.InferenceResponse(output_tensors=[embedding_tt])
            responses[i] = response

        return responses
