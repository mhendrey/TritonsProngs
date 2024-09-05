import torch
from transformers import SeamlessM4Tv2ForTextToText, SeamlessM4TProcessor


class SeamlessM4Tv2ForTextToTextMulti(SeamlessM4Tv2ForTextToText):
    """Redefines the `generate()` to allow for translating to different `tgt_lang`s.
    Otherwise it is the exact same.
    """

    def generate(
        self,
        input_ids=None,
        tgt_lang=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        **kwargs,
    ):
        """
        Generates sequences of token ids.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.Tensor` of varying shape depending on the modality, *optional*):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            tgt_lang (`str` | `List[str]`, *optional*):
                The language or languages to use as target language(s) for translation.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`. The possible
            [`~utils.ModelOutput`] types are:
                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # prepare text_decoder_input_ids
        text_decoder_input_ids = kwargs.pop("decoder_input_ids", None)
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        if tgt_lang is not None:
            batch_size = (
                len(input_ids)
                if input_ids is not None
                else len(kwargs.get("inputs_embeds"))
            )

            if hasattr(self.generation_config, "text_decoder_lang_to_code_id"):
                if isinstance(tgt_lang, str):
                    tgt_lang = [tgt_lang] * batch_size
                elif len(tgt_lang) != batch_size:
                    raise ValueError(
                        f"tgt_lang length, {len(tgt_lang)} != {batch_size} batch size"
                    )

                text_decoder_input_ids = []
                for tgt in tgt_lang:
                    # also accept __xxx__
                    tgt = tgt.replace("__", "")
                    if tgt not in self.generation_config.text_decoder_lang_to_code_id:
                        raise ValueError(
                            f"""`tgt_lang={tgt}` is not supported by this model. Please specify a `tgt_lang` in
                            {', '.join(self.generation_config.text_decoder_lang_to_code_id.keys())}"""
                        )
                    # tgt_lang gets priority over decoder input ids
                    text_tgt_lang_id = (
                        self.generation_config.text_decoder_lang_to_code_id.get(tgt)
                    )
                    text_decoder_input_ids.append(text_tgt_lang_id)

                text_decoder_input_ids = (
                    torch.tensor(text_decoder_input_ids).reshape(-1, 1).to(self.device)
                )
            else:
                raise ValueError(
                    """This model generation config doesn't have a `text_decoder_lang_to_code_id` key which maps
                    the target language to the right token id. Make sure to load the right generation config."""
                )
        else:
            # only a warning, otherwise errors appear in the tests
            print(
                """You must either specify a `tgt_lang` or pass a correct `text_decoder_input_ids` to get
                a correct generation, otherwise the generation will probably make no sense."""
            )

        return super(SeamlessM4Tv2ForTextToText, self).generate(
            input_ids,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            decoder_input_ids=text_decoder_input_ids,
            **kwargs,
        )


class SeamlessM4TProcessorMulti(SeamlessM4TProcessor):
    def __call__(self, text=None, audios=None, src_lang=None, tgt_lang=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to SeamlessM4TTokenizerFast's [`~SeamlessM4TTokenizerFast.__call__`] if `text` is not
        `None` to encode the text. To prepare the audio(s), this method forwards the `audios` and `kwargs` arguments to
        SeamlessM4TFeatureExtractor's [`~SeamlessM4TFeatureExtractor.__call__`] if `audios` is not `None`. Please refer
        to the doctsring of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audios (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case
                of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
                and T the sample length of the audio.
            src_lang (`str`, `List[str]`, *optional*):
                The language code(s) of the input texts/audios. If not specified, the last `src_lang` specified will be
                used.
            tgt_lang (`str`, *optional*):
                The code of the target language. If not specified, the last `tgt_lang` specified will be used.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to the feature extractor and/or the
                tokenizer.
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **input_features** -- Audio input features to be fed to a model. Returned when `audios` is not `None`.
        """
        if text is not None:
            # Functions as before
            if isinstance(src_lang, str):
                encoding = super().__call__(
                    text=text,
                    audios=audios,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    **kwargs,
                )
            elif isinstance(src_lang, list):
                if isinstance(text, str):
                    raise ValueError("Processor: `text` is str, but `src_lang` is list")
                if len(text) != len(src_lang):
                    raise ValueError(
                        f"Processor: `text` batch size != `src_lang` batch_size"
                    )
                encoding = super().__call__(
                    text=text,
                    audios=audios,
                    src_lang=src_lang[0],  # Just need the first one for now
                    tgt_lang=tgt_lang,
                    **kwargs,
                )
                src_lang_ids = self.tokenizer.convert_tokens_to_ids(
                    [f"__{src}__" for src in src_lang]
                )
                encoding["input_ids"][:, 0] = torch.LongTensor(src_lang_ids)
        # Audios functions as before
        else:
            encoding = super().__call__(
                text=text, audios=audios, src_lang=src_lang, tgt_lang=tgt_lang, **kwargs
            )

        return encoding
