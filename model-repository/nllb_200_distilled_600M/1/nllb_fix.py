import math
from types import NoneType
import torch
from transformers import NllbTokenizerFast, M2M100ForConditionalGeneration
from transformers.models.nllb.tokenization_nllb_fast import FAIRSEQ_LANGUAGE_CODES
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.utils import GenerationConfig, StoppingCriteriaList
from transformers.utils.generic import PaddingStrategy, TensorType
from transformers.tokenization_utils_base import TruncationStrategy
from typing import Callable, Union, Optional

# Used NllbTokenizerFast.convert_tokens_to_ids() with
# transformers.models.nllb.tokenization_nllb_fast.FAIRSEQ_LANGUAGE_CODES
# LANG_TOKEN_TO_ID = {}
# for lang_code in FAIRSEQ_LANGUAGE_CODES:
#   LANG_TOKEN_TO_ID[lang_code] = tokenizer.convert_tokens_to_ids(lang_code)
LANG_TOKEN_TO_ID = {
    "ace_Arab": 256001,
    "ace_Latn": 256002,
    "acm_Arab": 256003,
    "acq_Arab": 256004,
    "aeb_Arab": 256005,
    "afr_Latn": 256006,
    "ajp_Arab": 256007,
    "aka_Latn": 256008,
    "als_Latn": 256162,
    "amh_Ethi": 256009,
    "apc_Arab": 256010,
    "arb_Arab": 256011,
    "ars_Arab": 256012,
    "ary_Arab": 256013,
    "arz_Arab": 256014,
    "asm_Beng": 256015,
    "ast_Latn": 256016,
    "awa_Deva": 256017,
    "ayr_Latn": 256018,
    "azb_Arab": 256019,
    "azj_Latn": 256020,
    "bak_Cyrl": 256021,
    "bam_Latn": 256022,
    "ban_Latn": 256023,
    "bel_Cyrl": 256024,
    "bem_Latn": 256025,
    "ben_Beng": 256026,
    "bho_Deva": 256027,
    "bjn_Arab": 256028,
    "bjn_Latn": 256029,
    "bod_Tibt": 256030,
    "bos_Latn": 256031,
    "bug_Latn": 256032,
    "bul_Cyrl": 256033,
    "cat_Latn": 256034,
    "ceb_Latn": 256035,
    "ces_Latn": 256036,
    "cjk_Latn": 256037,
    "ckb_Arab": 256038,
    "crh_Latn": 256039,
    "cym_Latn": 256040,
    "dan_Latn": 256041,
    "deu_Latn": 256042,
    "dik_Latn": 256043,
    "dyu_Latn": 256044,
    "dzo_Tibt": 256045,
    "ell_Grek": 256046,
    "eng_Latn": 256047,
    "epo_Latn": 256048,
    "est_Latn": 256049,
    "eus_Latn": 256050,
    "ewe_Latn": 256051,
    "fao_Latn": 256052,
    "fij_Latn": 256054,
    "fin_Latn": 256055,
    "fon_Latn": 256056,
    "fra_Latn": 256057,
    "fur_Latn": 256058,
    "fuv_Latn": 256059,
    "gaz_Latn": 256135,
    "gla_Latn": 256060,
    "gle_Latn": 256061,
    "glg_Latn": 256062,
    "grn_Latn": 256063,
    "guj_Gujr": 256064,
    "hat_Latn": 256065,
    "hau_Latn": 256066,
    "heb_Hebr": 256067,
    "hin_Deva": 256068,
    "hne_Deva": 256069,
    "hrv_Latn": 256070,
    "hun_Latn": 256071,
    "hye_Armn": 256072,
    "ibo_Latn": 256073,
    "ilo_Latn": 256074,
    "ind_Latn": 256075,
    "isl_Latn": 256076,
    "ita_Latn": 256077,
    "jav_Latn": 256078,
    "jpn_Jpan": 256079,
    "kab_Latn": 256080,
    "kac_Latn": 256081,
    "kam_Latn": 256082,
    "kan_Knda": 256083,
    "kas_Arab": 256084,
    "kas_Deva": 256085,
    "kat_Geor": 256086,
    "kaz_Cyrl": 256089,
    "kbp_Latn": 256090,
    "kea_Latn": 256091,
    "khk_Cyrl": 256122,
    "khm_Khmr": 256092,
    "kik_Latn": 256093,
    "kin_Latn": 256094,
    "kir_Cyrl": 256095,
    "kmb_Latn": 256096,
    "kmr_Latn": 256099,
    "knc_Arab": 256087,
    "knc_Latn": 256088,
    "kon_Latn": 256097,
    "kor_Hang": 256098,
    "lao_Laoo": 256100,
    "lij_Latn": 256102,
    "lim_Latn": 256103,
    "lin_Latn": 256104,
    "lit_Latn": 256105,
    "lmo_Latn": 256106,
    "ltg_Latn": 256107,
    "ltz_Latn": 256108,
    "lua_Latn": 256109,
    "lug_Latn": 256110,
    "luo_Latn": 256111,
    "lus_Latn": 256112,
    "lvs_Latn": 256101,
    "mag_Deva": 256113,
    "mai_Deva": 256114,
    "mal_Mlym": 256115,
    "mar_Deva": 256116,
    "min_Latn": 256117,
    "mkd_Cyrl": 256118,
    "mlt_Latn": 256120,
    "mni_Beng": 256121,
    "mos_Latn": 256123,
    "mri_Latn": 256124,
    "mya_Mymr": 256126,
    "nld_Latn": 256127,
    "nno_Latn": 256128,
    "nob_Latn": 256129,
    "npi_Deva": 256130,
    "nso_Latn": 256131,
    "nus_Latn": 256132,
    "nya_Latn": 256133,
    "oci_Latn": 256134,
    "ory_Orya": 256136,
    "pag_Latn": 256137,
    "pan_Guru": 256138,
    "pap_Latn": 256139,
    "pbt_Arab": 256143,
    "pes_Arab": 256053,
    "plt_Latn": 256119,
    "pol_Latn": 256140,
    "por_Latn": 256141,
    "prs_Arab": 256142,
    "quy_Latn": 256144,
    "ron_Latn": 256145,
    "run_Latn": 256146,
    "rus_Cyrl": 256147,
    "sag_Latn": 256148,
    "san_Deva": 256149,
    "sat_Beng": 256150,
    "scn_Latn": 256151,
    "shn_Mymr": 256152,
    "sin_Sinh": 256153,
    "slk_Latn": 256154,
    "slv_Latn": 256155,
    "smo_Latn": 256156,
    "sna_Latn": 256157,
    "snd_Arab": 256158,
    "som_Latn": 256159,
    "sot_Latn": 256160,
    "spa_Latn": 256161,
    "srd_Latn": 256163,
    "srp_Cyrl": 256164,
    "ssw_Latn": 256165,
    "sun_Latn": 256166,
    "swe_Latn": 256167,
    "swh_Latn": 256168,
    "szl_Latn": 256169,
    "tam_Taml": 256170,
    "taq_Latn": 256177,
    "taq_Tfng": 256178,
    "tat_Cyrl": 256171,
    "tel_Telu": 256172,
    "tgk_Cyrl": 256173,
    "tgl_Latn": 256174,
    "tha_Thai": 256175,
    "tir_Ethi": 256176,
    "tpi_Latn": 256179,
    "tsn_Latn": 256180,
    "tso_Latn": 256181,
    "tuk_Latn": 256182,
    "tum_Latn": 256183,
    "tur_Latn": 256184,
    "twi_Latn": 256185,
    "tzm_Tfng": 256186,
    "uig_Arab": 256187,
    "ukr_Cyrl": 256188,
    "umb_Latn": 256189,
    "urd_Arab": 256190,
    "uzn_Latn": 256191,
    "vec_Latn": 256192,
    "vie_Latn": 256193,
    "war_Latn": 256194,
    "wol_Latn": 256195,
    "xho_Latn": 256196,
    "ydd_Hebr": 256197,
    "yor_Latn": 256198,
    "yue_Hant": 256199,
    "zho_Hans": 256200,
    "zho_Hant": 256201,
    "zsm_Latn": 256125,
    "zul_Latn": 256202,
}


class TgtLangIdsLogitsProcessor(LogitsProcessor):
    def __init__(self, bos_token_ids: list[int]):
        self.bos_token_ids = bos_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        batch_size = input_ids.shape[0]
        scores_processed = scores
        if cur_len == 1:
            scores_processed = torch.full_like(scores, -math.inf)
            for b, bos_token_id in enumerate(self.bos_token_ids):
                scores_processed[b, bos_token_id] = 0
        return scores_processed


class NllbMulti(M2M100ForConditionalGeneration):
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        tgt_lang: Union[str, list[str]] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], list[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        logits_processor = LogitsProcessorList()
        if tgt_lang is not None:
            if isinstance(tgt_lang, str):
                tgt_lang = [tgt_lang] * batch_size
            assert len(tgt_lang) == batch_size, (
                f"tgt_lang length, {len(tgt_lang)} " + f"does not match {batch_size=:}"
            )
            tgt_lang_ids = [LANG_TOKEN_TO_ID[tgt] for tgt in tgt_lang]
            logits_processor.append(TgtLangIdsLogitsProcessor(tgt_lang_ids))

        return super().generate(
            input_ids,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )


class NllbTokenizerFastMulti(NllbTokenizerFast):
    def __call__(
        self,
        text: Union[str, list[str], list[list[str]]] = None,
        src_lang: Union[str, list[str]] = None,
        text_pair: Union[str, list[str], list[list[str]], NoneType] = None,
        text_target: Union[str, list[str], list[list[str]]] = None,
        text_pair_target: Union[str, list[str], list[list[str]], NoneType] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Union[str, TensorType, NoneType] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        # Run tokenizer as before
        encoding = super().__call__(
            text=text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

        # Now fix first tokens if given a src_lang
        batch_size = len(encoding["input_ids"])
        if src_lang is not None:
            if isinstance(src_lang, str):
                src_lang = [src_lang] * batch_size
            else:
                assert len(src_lang) == batch_size, (
                    f"Length of src_lang list, "
                    + f"{len(src_lang)} must match {batch_size=:}"
                )

            src_lang_ids = self.convert_tokens_to_ids([src for src in src_lang])
            if return_tensors is None:
                for i, src_lang_id in enumerate(src_lang_ids):
                    encoding["input_ids"][i][0] = src_lang_id
            else:
                encoding["input_ids"][:, 0] = torch.LongTensor(src_lang_ids)

        return encoding
