import asyncio
import json
from transformers import AutoProcessor, AutoTokenizer

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for text embedding
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
        embed_models = json.loads(
            model_config["parameters"]["embed_models"]["string_value"]
        )
        self.processors = {}
        for embed_model, model_path in embed_models.items():
            if embed_model == "siglip_text":
                self.processors[embed_model] = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", local_files=True)
               
            elif embed_model == "multilingual_e5_large":
                self.processors[embed_model] = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large", local_files=True)
                
        ## Get additional parameters from the config.pbtxt file
        # Specify the default embedding model. Can be overriden in request parameter
        self.default_embed_model = model_config["parameters"]["default_embed_model"][
            "string_value"
        ]

    def process_request(self, request, embed_model: str):
        """
        Process the input text request and prepare input_ids for embedding.

        Parameters
        ----------
        request : pb_utils.InferenceRequest
            Inference request containing the input text.
        embed_model : str
            Embedding model to use.

        Returns
        -------
        np.ndarray
            Tokenized text
        """
        try:
            input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
        except Exception as exc:
            raise ValueError(f"Failed on getting input tensor from request: {exc}")

        try:
            input_text = [
                b.decode("utf-8") for b in input_text_tt.as_numpy().reshape(-1)
            ]
        except Exception as exc:
            raise ValueError(f"Failed on converting numpy to str request data: {exc}")

        try:
            input_ids_np = self.processors[embed_model](
                text=input_text, padding="max_length", return_tensors="pt"
            )["input_ids"].numpy()
            n_tokens = input_ids_np.shape[-1]

            # Safety Checks
            if embed_model == "multilingual_e5_large":
                # Could set truncation=True, but that seems dangerously silent for
                # something that could severely impact performance
                if n_tokens > 512:
                    raise ValueError(
                        f"Processing {input_text} has {n_tokens} tokens which exceeds max of 512."
                    )
                for text in input_text:
                    if not (text.startswith("query: ") or text.startswith("passage: ")):
                        raise ValueError(
                            f"Processing {text} must start with 'query: ' or"
                            + f"'passage: ' prefix when using {embed_model}"
                        )
            elif embed_model == "siglip_text":
                # Could set truncation=True, but that seems dangerously silent for
                # something that could severely impact performance
                if n_tokens > 64:
                    raise ValueError(
                        f"Processing {input_text} has {n_tokens} tokens which exceeds max of 64."
                    )
        except Exception as exc:
            raise ValueError(
                f"Failed on {embed_model}'s Processor(text=input_text): {exc}"
            )

        # Shape = [batch_size, 512 or 64], where batch_size should be 1
        return input_ids_np

    async def execute(self, requests: list) -> list:
        """
        Execute a batch of embedding requests on provided texts.

        Option Request Parameters
        -------------------------
        embed_model : str
            Specify which embedding model to use.
            If None, default_embed_model is used.

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]
            List of inference requests each containing text to be embedded.

        Returns
        -------
        List[pb_utils.InferenceResponse]
            List of response objects with embedding results or error messages
        """
        logger = pb_utils.Logger
        batch_size = len(requests)
        logger.log_info(f"embed_text.execute received {batch_size} requests")
        responses = [None] * batch_size
        inference_response_awaits = []
        valid_requests = []
        for batch_id, request in enumerate(requests):
            # Handle any request parameters
            request_params = json.loads(request.parameters())
            embed_model = request_params.get("embed_model", self.default_embed_model)

            if embed_model not in self.processors:
                responses[batch_id] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"{embed_model=:} not in {self.processors.keys()}"
                    )
                )
                continue

            try:
                input_ids_np = self.process_request(request, embed_model)
                input_ids_tt = pb_utils.Tensor("INPUT_IDS", input_ids_np)
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
                    inputs=[input_ids_tt],
                )
                # Perform asynchronous inference request
                inference_response_awaits.append(infer_model_request.async_exec())
                valid_requests.append(batch_id)

        inference_responses = await asyncio.gather(*inference_response_awaits)
        for model_response, batch_id in zip(inference_responses, valid_requests):
            if model_response.has_error() and responses[batch_id] is None:
                err_msg = (
                    "Error embedding the text: " + f"{model_response.error().message()}"
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
