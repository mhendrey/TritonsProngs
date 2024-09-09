import itertools
import json
import numpy as np
import torch
from typing import List

from nllb_fix import NllbTokenizerFastMulti, NllbMulti
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Perform translation using SeamlessM4T-large-v2's Text2Text"""

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        # Get TRANSLATED_TEXT configuration
        translated_text_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSLATED_TEXT"
        )
        # Convert Triton types to numpy types
        self.translated_text_dtype = pb_utils.triton_string_to_numpy(
            translated_text_config["data_type"]
        )

        # Use the GPU if available, otherwise use the CPU
        if args["model_instance_kind"] == "GPU" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch_dtype = torch.float16
            # attn_implementation = "flash_attention_2"
        else:
            self.device = torch.device("cpu")
            torch_dtype = torch.float32  # CPUs can't handle float16
            # attn_implementation = None
        self.model = NllbMulti.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            device_map="auto",
            torch_dtype=torch_dtype,
            local_files_only=True,
            # attn_implementation=attn_implementation,
        )
        self.tokenizer = NllbTokenizerFastMulti.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            local_files_only=True,
        )
        # Get list of supported language tokens. Of the form "eng_Latn"
        self.supported_languages = set(self.tokenizer.additional_special_tokens)

    def execute(self, requests: List) -> List:
        """
        Each request is sent by a client and represents appropriately chunked text
        for translation. The INPUT_TEXT is a 1-d array of with one element of bytes.
        The SRC_LANG and TGT_LANG inputs are 1-d array of bytes but have just one
        element in them.

        Parameters
        ----------
        requests : List[pb_utils.InferenceRequest]

        Returns
        -------
        responses: List[pb_utils.InferenceResponse]
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger = pb_utils.Logger
        batch_size = len(requests)
        logger.log_info(
            f"nllb_200_distilled_600M.execute received {batch_size} requests"
        )
        responses = [None] * batch_size
        valid_requests = []
        batch_input_text = []
        batch_src_lang = []
        batch_tgt_lang = []
        for batch_id, request in enumerate(requests):
            try:
                # Get the input data as Triton Tensors
                input_text_tt = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
                src_lang_tt = pb_utils.get_input_tensor_by_name(request, "SRC_LANG")
                tgt_lang_tt = pb_utils.get_input_tensor_by_name(request, "TGT_LANG")

                # Convert TritonTensor -> numpy -> python str
                # NOTE: Triton converts your input string to bytes so you need to decode
                input_text = [
                    b.decode("utf-8") for b in input_text_tt.as_numpy().reshape(-1)
                ]
                src_lang = [
                    b.decode("utf-8") for b in src_lang_tt.as_numpy().reshape(-1)
                ]
                tgt_lang = [
                    b.decode("utf-8") for b in tgt_lang_tt.as_numpy().reshape(-1)
                ]

                if self.unsupported_lang(src_lang[0]):
                    raise ValueError(
                        f"src_lang {src_lang[0]} is not supported by NLLB. Needs to "
                        + f"be one of {self.supported_languages}"
                    )
                if self.unsupported_lang(tgt_lang[0]):
                    raise ValueError(
                        f"tgt_lang {tgt_lang[0]} is not supported by NLLB. Needs to "
                        + f"be one of {self.supported_languages}"
                    )

                batch_input_text.append(input_text)
                batch_src_lang.append(src_lang)
                batch_tgt_lang.append(tgt_lang)
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"{exc}", pb_utils.TritonError.INVALID_ARG
                    )
                )
                responses[batch_id] = response
                continue
            else:
                valid_requests.append(batch_id)

        input_texts = list(itertools.chain.from_iterable(batch_input_text))
        src_langs = list(itertools.chain.from_iterable(batch_src_lang))
        tgt_langs = list(itertools.chain.from_iterable(batch_tgt_lang))
        # Run through the model for translation
        ## Tokenize
        try:
            input_ids = self.tokenizer(
                text=input_texts,
                src_lang=src_langs,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        except Exception as exc:
            # Error with the batch. Be careful error msg doesn't cross
            # contaminate user data
            for batch_id in valid_requests:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"nllb_200_distilled_600M.processor threw error tokenizing the batch: {exc}"
                    )
                )
                responses[batch_id] = response
            return responses

        ## Generate output tokens
        try:
            with torch.no_grad():
                output_tokens = self.model.generate(
                    **input_ids,
                    tgt_lang=tgt_langs,
                    num_beams=1,  # Massive throughput hit if > 1
                    num_return_sequences=1,
                    max_new_tokens=512,
                    no_repeat_ngram_size=3,
                )
        except Exception as exc:
            for batch_id in valid_requests:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"nllb_200_distilled_600M.model.generate threw error on batch: {exc}"
                    )
                )
                responses[batch_id] = response
            return responses

        ## Decode tokens to text
        try:
            translated_texts = self.tokenizer.batch_decode(
                output_tokens, skip_special_tokens=True
            )
        except Exception as exc:
            for batch_id in valid_requests:
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        "nllb_200_distilled_600M.processor.batch_decode threw on batch: {exc}"
                    )
                )
                responses[batch_id] = response
            return responses

        for batch_id, translated_text in zip(valid_requests, translated_texts):
            # Convert to TritonTensor & make the TritonInferenceResponse
            translated_text_tt = pb_utils.Tensor(
                "TRANSLATED_TEXT",
                np.array(translated_text, dtype=self.translated_text_dtype).reshape(
                    -1, 1
                ),
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[translated_text_tt],
            )
            responses[batch_id] = inference_response

        return responses

    def unsupported_lang(self, lang_id):
        if lang_id in self.supported_languages:
            return False
        else:
            return True
