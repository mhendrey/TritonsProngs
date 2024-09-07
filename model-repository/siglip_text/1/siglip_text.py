import json
import torch
from transformers import SiglipTextModel

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for SigLIP Text
    model.
    """

    def initialize(self, args):
        """
        Initialize SiglipTextModel and load configuration parameters. Using
        torch.compile() to speed up inference. The first few passes through the model
        may be delayed while torch.compile() does its magic.

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

        # Use the GPU if available, otherwise use the CPU
        if args["model_instance_kind"] == "GPU" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch_dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            torch_dtype = torch.float32  # CPUs can't handle float16

        self.model = SiglipTextModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            device_map="auto",
            torch_dtype=torch_dtype,
            local_files_only=True,
            use_safetensors=True,
        )
        # If on a GPU, use torch.compile to improve throughput
        if torch.cuda.is_available():
            self.model = torch.compile(self.model, dynamic=True)

    def execute(self, requests: list) -> list:
        """
        Execute a batch of embedding requests on provided texts that have already
        been converted to `input_ids`. When using
        `processor(text=prompts, padding='max_length')`, it will automatically pad
        with `1`.

        Shape = (64,), dtype=np.int64

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]
            List of inference requests each containing an image to be embedded.

        Returns
        -------
        List[pb_utils.InferenceResponse]
            List of response objects with embedding results or error messages
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger = pb_utils.Logger
        batch_size = len(requests)
        logger.log_info(f"siglip_text.execute received {batch_size} requests")
        responses = [None] * batch_size
        batch_input_ids = []
        valid_requests = []
        for batch_id, request in enumerate(requests):
            try:
                input_ids = torch.from_numpy(
                    pb_utils.get_input_tensor_by_name(
                        request,
                        "INPUT_IDS",
                    ).as_numpy()
                )
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"{exc}")
                )
                responses[batch_id] = response
            else:
                batch_input_ids.append(input_ids)
                valid_requests.append(batch_id)

        # Create batch to be processed shape=[len(valid_requests), 64]
        batch_input_ids = torch.cat(batch_input_ids, dim=0).to(self.device)
        try:
            with torch.no_grad():
                text_embedding_np = (
                    self.model(input_ids=batch_input_ids)["pooler_output"]
                    .cpu()
                    .type(torch.float32)
                    .numpy()
                )
        except Exception as exc:
            # Problem embedding the whole batch. They all failed
            for i in valid_requests:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        "siglip_text threw error embedding the batch. Check your "
                        + f"input text and/or try again. {exc}"
                    )
                )
                responses[i] = response
            return responses

        for i, embedding in zip(valid_requests, text_embedding_np):
            embedding_tt = pb_utils.Tensor("EMBEDDING", embedding.reshape(1, -1))
            response = pb_utils.InferenceResponse(output_tensors=[embedding_tt])
            responses[i] = response

        return responses
