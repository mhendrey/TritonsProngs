import asyncio
from collections import defaultdict
import json
import numpy as np
from typing import List

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Service Level Deployment Package
    This handles things nicely for clients. Taking in 'strings' [really means bytes]
    and then handles the logic for using the language id model (if not specified)
    before passing on to translation model."""

    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = model_config = json.loads(args["model_config"])

        # Get INPUT_TEXT configuration
        input_text_config = pb_utils.get_input_config_by_name(
            model_config, "INPUT_TEXT"
        )
        # Convert Triton types to numpy types
        self.input_text_dtype = pb_utils.triton_string_to_numpy(
            input_text_config["data_type"]
        )

        # Get TRANSLATED_TEXT configuration
        translated_text_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSLATED_TEXT"
        )
        # Convert Triton types to numpy types
        self.translated_text_dtype = pb_utils.triton_string_to_numpy(
            translated_text_config["data_type"]
        )

        # Get default values
        self.default_language_id_model = model_config["parameters"][
            "default_language_id_model"
        ]["string_value"]
        self.default_sentence_segmenter = model_config["parameters"][
            "default_sentence_segmenter"
        ]["string_value"]
        self.default_translation_model = model_config["parameters"][
            "default_translation_model"
        ]["string_value"]
        self.default_language_id_threshold = float(
            model_config["parameters"]["default_language_id_threshold"]["string_value"]
        )

        # Batch convenient collections
        self.responses = [None] * 0
        self.is_ok = [True] * 0

    def reset_responses_is_ok(self, batch_size: int):
        self.responses = [None] * batch_size
        self.is_ok = [True] * batch_size

    def process_request_data(self, requests: list, requests_data: dict) -> None:
        """_summary_

        Parameters
        ----------
        requests : list
            _description_
        responses : list
            _description_
        is_ok: list
            _description_
        requests_data: dict
            _description_

        Returns
        -------
        """
        for batch_id, request in enumerate(requests):
            try:
                input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"{exc}", pb_utils.TritonError.INVALID_ARG
                    )
                )
                if self.responses[batch_id] is None:
                    self.responses[batch_id] = response
                self.is_ok[batch_id] = False
                continue
            requests_data[batch_id]["input_text_tt"] = input_text_tt

            # Get any optional parameters passed in.
            request_params = json.loads(request.parameters())
            ## Pipeline Stages
            requests_data[batch_id]["language_id_model"] = request_params.get(
                "language_id_model", self.default_language_id_model
            )
            requests_data[batch_id]["sentence_segmenter"] = request_params.get(
                "sentence_segmenter", self.default_sentence_segmenter
            )
            requests_data[batch_id]["translation_model"] = request_params.get(
                "translation_model", self.default_translation_model
            )
            ## src_lang
            requests_data[batch_id]["src_lang"] = request_params.get("src_lang", None)
            ## tgt_lang, planning for adding NLLB which has different codes
            if requests_data[batch_id]["translation_model"] == "seamlessm4t_text2text":
                default_tgt_lang = "eng"
            elif requests_data[batch_id]["translation_model"] == "nllb":
                default_tgt_lang = "eng_Latn"
            requests_data[batch_id]["tgt_lang"] = request_params.get(
                "tgt_lang", default_tgt_lang
            )
            ## Language ID Threshold
            requests_data[batch_id]["language_id_threshold"] = request_params.get(
                "language_id_threshold", self.default_language_id_threshold
            )

        return None

    def submit_inference_request(
        self, model_name: str, requested_output_names: list, inputs_tt: list
    ):
        # logger = pb_utils.Logger
        try:
            infer_request = pb_utils.InferenceRequest(
                model_name=model_name,
                requested_output_names=requested_output_names,
                inputs=inputs_tt,
            )
        except Exception as exc:
            self.logger.log_error(f"{exc}")
        return infer_request

    def get_inference_response(
        self,
        infer_response,
        batch_id: int,
        requested_output_names: list,
        error_msg: str = "",
    ) -> list:
        # logger = pb_utils.Logger
        if infer_response.has_error():
            error_msg += f" {batch_id=:} threw {infer_response.error().message()}"
            self.error_response(batch_id, error_msg)
            raise RuntimeError()
        else:
            outputs_tt = []
            for output_name in requested_output_names:
                try:
                    output_tt = pb_utils.get_output_tensor_by_name(
                        infer_response, output_name
                    )
                except Exception as exc:
                    self.logger.log_error(f"{output_name=:} threw {exc}")
                outputs_tt.append(output_tt)
            return outputs_tt

    def seamless_fix_chinese(self, src_lang_tt, src_script_tt):
        src_lang = src_lang_tt.as_numpy().reshape(-1)[0].decode("utf-8")
        if src_script_tt is not None:
            src_script = src_script_tt.as_numpy().reshape(-1)[0].decode("utf-8")
        else:
            src_script = ""
        if src_lang == "zho":
            if src_script == "Hant":
                src_lang = "cmn_Hant"
            else:
                src_lang = "cmn"
            return pb_utils.Tensor(
                "SRC_LANG", np.array([src_lang], dtype=np.object_).reshape(-1, 1)
            )
        else:
            return src_lang_tt

    def error_response(self, batch_id: int, error_msg: str):
        response = pb_utils.InferenceResponse(error=pb_utils.TritonError(error_msg))
        if self.responses[batch_id] is None:
            self.responses[batch_id] = response
        self.is_ok[batch_id] = False
        self.logger.log_error(error_msg)

    async def execute(self, requests: List) -> List:
        """
        Each request is one document that a client has submitted for translation

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]
            List of request submitted by clients. In this simple example, this should
            have a length of just one since dynamic batching is not enabled.

        Returns
        -------
        List[pb_utils.InferenceResponse]
            Each response is the translated document for a given client's request
        """
        logger = pb_utils.Logger
        batch_size = len(requests)
        logger.log_info(f"`translate` received {batch_size} requests in dynamic batch")
        self.reset_responses_is_ok(batch_size)
        requests_data = defaultdict(dict)
        # requests_data = {
        # ----<batch_id>: {
        # --------"input_text_tt": input_text_tt,
        # --------"src_lang": None,
        # --------"tgt_lang": "eng",
        # --------"language_id_threshold": 0.95,
        # --------"language_id_model": "fasttext_language_identification",
        # --------"sentence_segmenter": "sentencex",
        # --------"translation_model": "seamlessm4t_text2text"
        # ----}
        # }
        translate_inputs = defaultdict(dict)
        # translate_inputs = {
        # ----<batch_id>: {
        # --------<chunk_id>: {
        # ------------"input_text_tt": input_text_tt,
        # ------------"src_lang_tt": src_lang_tt,
        # ------------"tgt_lang_tt": tgt_lang_tt,
        # --------}
        # ----}
        # }

        # Get input data and request parameters for all requests
        self.process_request_data(requests, requests_data)

        # Submit valid requests for document level identification. Needed for
        # sentence segmentation
        # **NOTE** See if pb_utils.InferenceResponse(warnings=) is a thing
        # Would be nice to send a warning back if client provided src_lang
        # but it conflicted with language id model. Don't want to override
        # a client since the language id model could be wrong.
        doc_lang_await = []
        doc_lang_batch_ids = []
        src_lang_doc_tts = {}
        src_script_doc_tts = {}
        prob_docs = {}
        for batch_id, request_data in requests_data.items():
            # Should all be ok at this point
            if not self.is_ok[batch_id]:
                continue
            src_lang = request_data["src_lang"]
            if src_lang:
                src_lang_doc_tts[batch_id] = pb_utils.Tensor(
                    "SRC_LANG",
                    np.array([src_lang.encode("utf-8")], np.object_).reshape(-1, 1),
                )
                src_script_doc_tts[batch_id] = None
                prob_docs[batch_id] = 1.0
            else:
                doc_lang_batch_ids.append(batch_id)
                doc_lang_await.append(
                    self.submit_inference_request(
                        model_name=request_data["language_id_model"],
                        requested_output_names=[
                            "SRC_LANG",
                            "SRC_SCRIPT",
                            "PROBABILITY",
                        ],
                        inputs_tt=[request_data["input_text_tt"]],
                    ).async_exec()
                )

        # Now submit those with a src_lang to be split into sentences
        # While the language_id_model works on those without.
        sentence_segmenter_await = []
        sentence_segmenter_batch_ids = []
        for batch_id, src_lang_doc_tt in src_lang_doc_tts.items():
            if not self.is_ok[batch_id]:
                continue
            sentence_segmenter_batch_ids.append(batch_id)
            sentence_segmenter_await.append(
                self.submit_inference_request(
                    model_name=requests_data[batch_id]["sentence_segmenter"],
                    requested_output_names=["SENTENCES"],
                    inputs_tt=[
                        src_lang_doc_tt,
                        requests_data[batch_id]["input_text_tt"],
                    ],
                ).async_exec()
            )

        # Wait for language detection on the docs to finish
        doc_lang_responses = await asyncio.gather(*doc_lang_await)
        for batch_id, doc_response in zip(doc_lang_batch_ids, doc_lang_responses):
            try:
                src_lang_doc_tt, src_script_doc_tt, prob_doc_tt = (
                    self.get_inference_response(
                        doc_response,
                        batch_id,
                        requested_output_names=[
                            "SRC_LANG",
                            "SRC_SCRIPT",
                            "PROBABILITY",
                        ],
                        error_msg=f"{requests_data[batch_id]['language_id_model']}",
                    )
                )
            except Exception as exc:
                self.error_response(
                    batch_id, f"Gathering doc_lang_responses threw {exc}"
                )
            else:
                src_lang_doc_tts[batch_id] = src_lang_doc_tt
                src_script_doc_tts[batch_id] = src_script_doc_tt
                prob_docs[batch_id] = prob_doc_tt.as_numpy().reshape(-1)[0]

        # Submit these for sentence segmentation now too
        for batch_id in doc_lang_batch_ids:
            if not self.is_ok[batch_id]:
                continue
            src_lang_doc_tt = src_lang_doc_tts[batch_id]
            input_text_tt = requests_data[batch_id]["input_text_tt"]
            sentence_segmenter_batch_ids.append(batch_id)
            sentence_segmenter_await.append(
                self.submit_inference_request(
                    model_name=requests_data[batch_id]["sentence_segmenter"],
                    requested_output_names=["SENTENCES"],
                    inputs_tt=[src_lang_doc_tt, input_text_tt],
                ).async_exec()
            )

        # Await for all the sentence splitting
        translate_inputs = defaultdict(dict)
        sentence_segmenter_responses = await asyncio.gather(*sentence_segmenter_await)
        for batch_id, sentences_response in zip(
            sentence_segmenter_batch_ids, sentence_segmenter_responses
        ):
            if not self.is_ok[batch_id]:
                continue
            try:
                (sentences_tt,) = self.get_inference_response(
                    sentences_response,
                    batch_id,
                    requested_output_names=["SENTENCES"],
                    error_msg=f"{requests_data[batch_id]['sentence_segmenter']}",
                )
                for chunk_id, sentence in enumerate(
                    sentences_tt.as_numpy().reshape(-1)
                ):
                    translate_inputs[batch_id][chunk_id] = {
                        "input_text_tt": pb_utils.Tensor(
                            "INPUT_TEXT",
                            np.array([sentence], dtype=np.object_).reshape(-1, 1),
                        )
                    }
            except Exception as exc:
                self.error_response(
                    f"Gathering sentence_segmenter_responses threw {exc}"
                )
        # For those that have prob_doc < langauge_id_threshold, we need to do language
        # identification for each of the sentences. Let's submit those now
        sentence_lang_id_await = []
        sentence_lang_id_batch_chunk_ids = []
        translate_await = []
        translate_batch_chunk_ids = []
        for batch_id in translate_inputs:
            if not self.is_ok[batch_id]:
                continue
            if prob_docs[batch_id] < requests_data[batch_id]["language_id_threshold"]:
                for chunk_id in translate_inputs[batch_id]:
                    sentence_tt = translate_inputs[batch_id][chunk_id]["input_text_tt"]
                    sentence_lang_id_batch_chunk_ids.append((batch_id, chunk_id))
                    sentence_lang_id_await.append(
                        self.submit_inference_request(
                            model_name=requests_data[batch_id]["language_id_model"],
                            requested_output_names=["SRC_LANG", "SRC_SCRIPT"],
                            inputs_tt=[sentence_tt],
                        ).async_exec()
                    )
            else:
                # Submit these for translation
                for chunk_id in translate_inputs[batch_id]:
                    sentence_tt = translate_inputs[batch_id][chunk_id]["input_text_tt"]
                    src_lang_tt = src_lang_doc_tts[batch_id]
                    src_script_tt = src_script_doc_tts[batch_id]
                    if (
                        requests_data[batch_id]["translation_model"]
                        == "seamlessm4t_text2text"
                    ):
                        src_lang_tt = self.seamless_fix_chinese(
                            src_lang_tt, src_script_tt
                        )
                    tgt_lang_tt = pb_utils.Tensor(
                        "TGT_LANG",
                        np.array(
                            [requests_data[batch_id]["tgt_lang"]], dtype=np.object_
                        ).reshape(-1, 1),
                    )
                    translate_batch_chunk_ids.append((batch_id, chunk_id))
                    translate_await.append(
                        self.submit_inference_request(
                            model_name=requests_data[batch_id]["translation_model"],
                            requested_output_names=["TRANSLATED_TEXT"],
                            inputs_tt=[sentence_tt, src_lang_tt, tgt_lang_tt],
                        ).async_exec()
                    )
        # Await for language detection at sentence level and gather results and
        # submit these for translation
        sentence_lang_id_responses = await asyncio.gather(*sentence_lang_id_await)
        for (batch_id, chunk_id), sentence_lang_id_response in zip(
            sentence_lang_id_batch_chunk_ids, sentence_lang_id_responses
        ):
            try:
                src_lang_tt, src_script_tt = self.get_inference_response(
                    sentence_lang_id_response,
                    batch_id,
                    requested_output_names=["SRC_LANG", "SRC_SCRIPT"],
                    error_msg=f"{requests_data[batch_id]['language_id_model']} "
                    + "on sentences",
                )
                sentence_tt = translate_inputs[batch_id][chunk_id]["input_text_tt"]
                if (
                    requests_data[batch_id]["translation_model"]
                    == "seamlessm4t_text2text"
                ):
                    src_lang_tt = self.seamless_fix_chinese(src_lang_tt, src_script_tt)
                tgt_lang_tt = pb_utils.Tensor(
                    "TGT_LANG",
                    np.array(
                        [requests_data[batch_id]["tgt_lang"]], dtype=np.object_
                    ).reshape(-1, 1),
                )
                translate_batch_chunk_ids.append((batch_id, chunk_id))
                translate_await.append(
                    self.submit_inference_request(
                        model_name=requests_data[batch_id]["translation_model"],
                        requested_output_names=["TRANSLATED_TEXT"],
                        inputs_tt=[sentence_tt, src_lang_tt, tgt_lang_tt],
                    ).async_exec()
                )
            except Exception as exc:
                self.error_response(
                    batch_id,
                    f"Gathering sentence-level lang_id for translation threw {exc}",
                )

        # Gather the translation results
        translate_responses = await asyncio.gather(*translate_await)

        results = defaultdict(dict)
        for (batch_id, chunk_id), translate_response in zip(
            translate_batch_chunk_ids, translate_responses
        ):
            try:
                (translated_chunk_tt,) = self.get_inference_response(
                    translate_response,
                    batch_id,
                    requested_output_names=["TRANSLATED_TEXT"],
                    error_msg=f"{requests_data[batch_id]['translation_model']} threw",
                )
                translated_chunk = (
                    translated_chunk_tt.as_numpy().reshape(-1)[0].decode("utf-8")
                )
                results[batch_id][chunk_id] = translated_chunk
            except Exception as exc:
                self.error_response(
                    batch_id, f"Gathering translated results threw {exc}"
                )

        for batch_id in sorted(results):
            if self.is_ok[batch_id]:
                result = results[batch_id]
                translated_chunks = [result[chunk_id] for chunk_id in sorted(result)]
                translated_doc = " ".join(translated_chunks)
                translated_doc_tt = pb_utils.Tensor(
                    "TRANSLATED_TEXT",
                    np.array([translated_doc], dtype=self.translated_text_dtype),
                )
                # Create the response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[translated_doc_tt]
                )
                self.responses[batch_id] = inference_response

        return self.responses
