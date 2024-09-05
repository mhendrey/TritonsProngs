import json
import numpy as np
from sentencex import segment

from iso_639_3_1 import ISO_639_3_1
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for splitting a
    string of text into sentences. Needed because nearly all machine translations
    expect to have sentences to translate....don't get me started.
    """

    def initialize(self, args):
        """
        Initialize any items needed later.

        Parameters
        ----------
        args : dict
            Command-line arguments for launching Triton Inference Server
        """
        self.model_config = model_config = json.loads(args["model_config"])
        sentences_config = pb_utils.get_output_config_by_name(model_config, "SENTENCES")
        self.sentences_dtype = pb_utils.triton_string_to_numpy(
            sentences_config["data_type"]
        )

    def execute(self, requests: list) -> list:
        """
        Execute a splitting a batch of requests into sentences.

        Parameters
        ----------
        requests : list[pb_utils.InferenceRequest]
            List of inference requests each containing text to be split

        Returns
        -------
        list[pb_utils.InferenceResponse]
            List of response objects containing array of sentences or error messages
        """
        logger = pb_utils.Logger
        batch_size = len(requests)
        logger.log_info(f"sentencex received {batch_size} requests")
        responses = [None] * batch_size
        for batch_id, request in enumerate(requests):
            # Get SRC_LANG & INPUT_TEXT from the request as Triton Tensors
            try:
                lang_id_tt = pb_utils.get_input_tensor_by_name(request, "SRC_LANG")
                input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"{exc}", pb_utils.TritonError.INVALID_ARG
                    )
                )
                responses[batch_id] = response
                continue

            # Convert Triton Tensors, both TYPE_STRING, to numpy (dtype=np.object_)
            # TYPE_STRING is bytes when sending a request. Decode to get str
            input_text = input_text_tt.as_numpy().reshape(-1)[0].decode("utf-8")
            lang_id = lang_id_tt.as_numpy().reshape(-1)[0].decode("utf-8")

            # In sentencex.segment(lang_id, text), lang_id: str, language identifier in
            # ISO 639-1 format. fastText gives ISO 639-3. So try to convert from 3 -> 1
            # if possible. Otherwise leave it alone
            # If lang_id is ISO 639-3, convert to ISO 639-1. Otherwise leave alone.
            lang_id = ISO_639_3_1.get(lang_id, lang_id)

            # Run through the sentencex.segment
            try:
                sentences = list(segment(lang_id, input_text))
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"sentencex threw {exc}")
                )
                responses[batch_id] = response
                continue

            # Make Triton Inference Response
            sentences_tt = pb_utils.Tensor(
                "SENTENCES",
                np.array(sentences, dtype=self.sentences_dtype).reshape(1, -1),
            )
            response = pb_utils.InferenceResponse(output_tensors=[sentences_tt])
            responses[batch_id] = response

        return responses
