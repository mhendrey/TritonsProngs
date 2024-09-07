import json
import torch
from transformers import AutoModel

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for the
    Multilingual-e5-Large text embedding model.
    """

    @staticmethod
    def average_pool(
        last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Taken directly from the Huggingface Model Card

        Parameters
        ----------
        last_hidden_states : torch.Tensor
            _description_
        attention_mask : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def initialize(self, args):
        """
        Initialize Multilingual-e5-large and load configuration parameters. Using
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

        self.pad_value = 1
        self.model = AutoModel.from_pretrained(
            "intfloat/multilingual-e5-large",
            device_map="auto",
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=True,
        )
        # If on a GPU, use torch.compile to improve throughput
        if torch.cuda.is_available():
            self.model = torch.compile(self.model, dynamic=True)

    def execute(self, requests: list) -> list:
        """
        Execute a batch of embedding requests on provided texts that have already
        been converted to `input_ids`. When using the Tokenizer, set
        padding='max_length'.

        **NOTE** We may need to pass in attention mask too. Or construct it on the fly.

        Shape = (512,), dtype=np.int64

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
        logger.log_info(f"multilingual-e5-large.execute received {batch_size} requests")
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

        # Create batch to be processed shape=[len(valid_requests), 512]
        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        attention_mask = (batch_input_ids != self.pad_value).type(torch.int64)
        batch_input_ids = batch_input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_input_ids, attention_mask=attention_mask
                )
                embeddings = (
                    TritonPythonModel.average_pool(
                        outputs.last_hidden_state, attention_mask
                    )
                    .cpu()
                    .type(torch.float32)
                )
                text_embedding_np = torch.nn.functional.normalize(
                    embeddings, p=2, dim=1
                ).numpy()
        except Exception as exc:
            # Problem embedding the whole batch. They all failed
            for i in valid_requests:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        "multilingual-e5-large threw error embedding the batch. Check your "
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
