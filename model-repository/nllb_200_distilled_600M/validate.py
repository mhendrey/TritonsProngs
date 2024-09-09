from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import json
import requests
from sacrebleu.metrics import CHRF


def get_translations(
    texts: list[str], src_langs: list[str], tgt_langs: list[str], max_workers: int = 50
) -> list[str]:
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, (text, src_lang, tgt_lang) in enumerate(
            zip(texts, src_langs, tgt_langs)
        ):
            inference_request = {
                "inputs": [
                    {
                        "name": "INPUT_TEXT",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": [text],
                    },
                    {
                        "name": "SRC_LANG",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": [src_lang],
                    },
                    {
                        "name": "TGT_LANG",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": [tgt_lang],
                    },
                ]
            }
            future = executor.submit(
                requests.post,
                url="http://localhost:8000/v2/models/nllb_200_distilled_600M/infer",
                json=inference_request,
            )
            futures[future] = i
        # Wait for results to come back
        for future in as_completed(futures):
            i = futures[future]
            try:
                response_json = future.result().json()
            except Exception as exc:
                raise ValueError(f"{texts[i]} threw {exc}")

            try:
                translated_text = response_json["outputs"][0]["data"][0]
            except:
                raise ValueError(f"{texts[i]} threw {response_json}")
            results[i] = translated_text
    return results


def test_pair(src, tgt):
    flores = load_dataset("facebook/flores", "all", split="devtest")
    chrf = CHRF(word_order=2, eps_smoothing=True)
    if src == "cmn_Hant":
        flores_src = "zho_Hant"
    elif src == "cmn":
        flores_src = "zho_Hans"
    else:
        flores_src = src
    src_sentence = [
        c for c in flores.column_names if c.startswith(f"sentence_{flores_src}")
    ]
    if src_sentence:
        src_sentence = src_sentence[0]
    else:
        raise ValueError(f"{src=:} {flores_src=:} not in flores")

    if tgt == "cmn_Hant":
        flores_tgt = "zho_Hant"
    elif tgt == "cmn":
        flores_tgt = "zho_Hans"
    else:
        flores_tgt = tgt
    tgt_sentence = [
        c for c in flores.column_names if c.startswith(f"sentence_{flores_tgt}")
    ]
    if tgt_sentence:
        tgt_sentence = tgt_sentence[0]
    else:
        raise ValueError(f"{tgt=:} {flores_tgt=:} not in flores")

    tgt_texts = []
    translations = []
    for batch in flores.iter(batch_size=60):
        n_batch = len(batch["id"])
        src_langs = [src] * n_batch
        tgt_langs = [tgt] * n_batch
        texts = batch[src_sentence]
        tgt_texts += batch[tgt_sentence]
        for t in get_translations(texts, src_langs, tgt_langs):
            translations.append(t)

    return chrf.corpus_score(translations, [tgt_texts]).score


def main():
    # These are the valid language codes in SeamlessM4Tv2Large
    # Notice that zho is not here. Need to rename that to cmn
    language_codes = [
        "afr_Latn",
        "amh_Ethi",
        "arb_Arab",
        "ary_Arab",
        "arz_Arab",
        "asm_Beng",
        "azj_Latn",
        "bel_Cyrl",
        "ben_Beng",
        "bos_Latn",
        "bul_Cyrl",
        "cat_Latn",
        "ceb_Latn",
        "ces_Latn",
        "ckb_Arab",
        "cym_Latn",
        "dan_Latn",
        "deu_Latn",
        "ell_Grek",
        # "eng_Latn", # Skip English
        "est_Latn",
        "eus_Latn",
        "fin_Latn",
        "fra_Latn",
        "fuv_Latn",
        "gaz_Latn",
        "gle_Latn",
        "glg_Latn",
        "guj_Gujr",
        "heb_Hebr",
        "hin_Deva",
        "hrv_Latn",
        "hun_Latn",
        "hye_Armn",
        "ibo_Latn",
        "ind_Latn",
        "isl_Latn",
        "ita_Latn",
        "jav_Latn",
        "jpn_Jpan",
        "kan_Knda",
        "kat_Geor",
        "kaz_Cyrl",
        "khk_Cyrl",
        "khm_Khmr",
        "kir_Cyrl",
        "kor_Hang",
        "lao_Laoo",
        "lit_Latn",
        "lug_Latn",
        "luo_Latn",
        "lvs_Latn",
        "mai_Deva",
        "mal_Mlym",
        "mar_Deva",
        "mkd_Cyrl",
        "mlt_Latn",
        "mni_Beng",
        "mya_Mymr",
        "nld_Latn",
        "nno_Latn",
        "nob_Latn",
        "npi_Deva",
        "nya_Latn",
        "ory_Orya",
        "pan_Guru",
        "pbt_Arab",
        "pes_Arab",
        "pol_Latn",
        "por_Latn",
        "ron_Latn",
        "rus_Cyrl",
        # "sat_Beng", # Flores has sat_Olck, but not sat_Beng
        "slk_Latn",
        "slv_Latn",
        "sna_Latn",
        "snd_Arab",
        "som_Latn",
        "spa_Latn",
        "srp_Cyrl",
        "swe_Latn",
        "swh_Latn",
        "tam_Taml",
        "tel_Telu",
        "tgk_Cyrl",
        "tgl_Latn",
        "tha_Thai",
        "tur_Latn",
        "ukr_Cyrl",
        "urd_Arab",
        "uzn_Latn",
        "vie_Latn",
        "yor_Latn",
        "yue_Hant",
        "zho_Hans",
        "zho_Hant",
        "zsm_Latn",
        "zul_Latn",
    ]

    errors = []
    chrf2 = []
    print(f"| Language | chrF2++ |")
    print(f"| :------: | :-----: |")
    for src in language_codes:
        try:
            triton_score = test_pair(src, "eng_Latn")
            chrf2.append(triton_score)
        except Exception as exc:
            errors.append((src, exc))
            continue
        print(f"| {src} | {triton_score:.1f} |")

    mean_score = sum(chrf2) / len(chrf2)
    print(f"| **Mean** | **{mean_score:.2f}**")
    print(f"\n\nMean = {mean_score:.2f}")

    for src, exc in errors:
        print(f"{src} threw {exc}")


if __name__ == "__main__":
    main()
