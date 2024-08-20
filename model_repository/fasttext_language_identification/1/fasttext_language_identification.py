import fasttext
import json
import numpy as np
import re
from typing import List

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """This model uses fasttext to perform language identification"""

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        # Get output configs
        src_lang_config = pb_utils.get_output_config_by_name(model_config, "SRC_LANG")
        src_script_config = pb_utils.get_output_config_by_name(
            model_config, "SRC_SCRIPT"
        )
        probability_config = pb_utils.get_output_config_by_name(
            model_config, "PROBABILITY"
        )

        # Convert Triton types to numpy types for output data
        self.src_lang_dtype = pb_utils.triton_string_to_numpy(
            src_lang_config["data_type"]
        )
        self.src_script_dtype = pb_utils.triton_string_to_numpy(
            src_script_config["data_type"]
        )
        self.probability_dtype = pb_utils.triton_string_to_numpy(
            probability_config["data_type"]
        )

        # Get parameters from config.pbtext
        model_path = model_config["parameters"]["model_path"]["string_value"]
        self.default_top_k = int(
            model_config["parameters"]["default_top_k"]["string_value"]
        )
        self.default_threshold = float(
            model_config["parameters"]["default_threshold"]["string_value"]
        )

        self.model = fasttext.load_model(model_path)
        self.REMOVE_NEWLINE = re.compile(r"\n")

    def execute(self, requests: List) -> List:
        """Predict the language id of the text provided in the request. Newlines are
        stripped since they throw an error.

        Default behavior has `top_k` = 1 and `threshold` = 0.0. You can set the `top_k`
        and `threshold` via request parameters to enable returning more than the top
        result and use the threshold to only return predictions whose probability
        exceeds the threshold value.

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]
            Input must contain the INPUT_TEXT

        Returns
        -------
        List[pb_utils.InferenceResponse]
        """
        logger = pb_utils.Logger
        batch_size = len(requests)
        logger.log_info(
            f"fasttext_language_identification received {batch_size} requests"
        )
        responses = [None] * batch_size
        for batch_id, request in enumerate(requests):
            # Handle any request parameters
            request_params = json.loads(request.parameters())
            top_k = request_params.get("top_k", self.default_top_k)
            threshold = request_params.get("threshold", self.default_threshold)

            # Get INPUT_TEXT from request. This is a Triton Tensor
            try:
                input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"{exc}", pb_utils.TritonError.INVALID_ARG
                    )
                )
                responses[batch_id] = response
                continue

            # Convert Triton Tensor (TYPE_STRING) to numpy (dtype=np.object_)
            # Array has just one element (config.pbtxt has dims: [1])
            # TYPE_STRING is bytes when sending through a request. Decode to get str
            input_text = input_text_tt.as_numpy().reshape(-1)[0].decode("utf-8")
            # Replace newlines with ' '. FastText breaks on \n
            input_text_cleaned = self.REMOVE_NEWLINE.sub(" ", input_text)

            # Run through the model
            try:
                output_labels, probs = self.model.predict(
                    input_text_cleaned, k=top_k, threshold=threshold
                )
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"{exc}")
                )
                responses[batch_id] = response
                continue

            src_langs = []
            src_scripts = []
            for output_label in output_labels:
                # Returns '__label__<lang_id>_<script>', e.g., '__label__spa_Latn'
                src_lang, src_script = output_label.replace("__label__", "").split("_")
                src_langs.append(src_lang)
                src_scripts.append(src_script)

            # Make Triton Inference Response
            src_lang_tt = pb_utils.Tensor(
                "SRC_LANG",
                np.array(src_langs, dtype=self.src_lang_dtype).reshape(1, -1),
            )
            src_script_tt = pb_utils.Tensor(
                "SRC_SCRIPT",
                np.array(src_scripts, dtype=self.src_script_dtype).reshape(1, -1),
            )
            probability_tt = pb_utils.Tensor("PROBABILITY", probs.reshape(1, -1))
            response = pb_utils.InferenceResponse(
                output_tensors=[src_lang_tt, src_script_tt, probability_tt],
            )
            responses[batch_id] = response

        return responses
