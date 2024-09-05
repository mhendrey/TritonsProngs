from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import numpy as np
from pprint import pprint
import requests
from sacrebleu.metrics import CHRF


def get_translations(
    texts: list[str], src_langs: list[str], tgt_langs: list[str], max_workers: int = 50
) -> list[str]:
    results = [None] * len(texts)
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, (text, src_lang, tgt_lang) in enumerate(
            zip(texts, src_langs, tgt_langs)
        ):
            inference_request = {
                "parameters": {"tgt_lang": tgt_lang},
                "inputs": [
                    {
                        "name": "INPUT_TEXT",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": [text],
                    },
                ],
            }
            if src_lang:
                inference_request["parameters"]["src_lang"] = src_lang
            future = executor.submit(
                requests.post,
                url="http://localhost:8000/v2/models/translate/infer",
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
                errors.append(f"{texts[i]} threw {response_json}")
                # raise ValueError(f"{texts[i]} threw {response_json}")
            else:
                results[i] = translated_text
    return results, errors


def test_pair(src, tgt, use_src: bool = True):
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
    errors = defaultdict(list)
    for batch in flores.iter(batch_size=60):
        texts = []
        src_langs = []
        tgt_langs = []
        for text_chunk in np.array_split(batch[src_sentence], 3):
            texts.append(" ".join(text_chunk))
            if use_src:
                src_langs.append(src)
            else:
                src_langs.append(None)
            tgt_langs.append(tgt)
        for text_chunk in np.array_split(batch[tgt_sentence], 3):
            tgt_texts.append(" ".join(text_chunk))
        results, errs = get_translations(texts, src_langs, tgt_langs)
        if errs:
            errors[src] += errs
        for t in results:
            if t is not None:
                translations.append(t)

    return chrf.corpus_score(translations, [tgt_texts]).score, errors


def main():
    # These are the valid language codes in SeamlessM4Tv2Large
    # Notice that zho is not here. Need to rename that to cmn
    language_codes = [
        "afr",
        "amh",
        "arb",
        "ary",
        "arz",
        "asm",
        "azj",
        "bel",
        "ben",
        "bos",
        "bul",
        "cat",
        "ceb",
        "ces",
        "ckb",
        "cmn",
        "cmn_Hant",
        "cym",
        "dan",
        "deu",
        "ell",
        # "eng", # Skip English
        "est",
        "eus",
        "fin",
        "fra",
        "fuv",
        "gaz",
        "gle",
        "glg",
        "guj",
        "heb",
        "hin",
        "hrv",
        "hun",
        "hye",
        "ibo",
        "ind",
        "isl",
        "ita",
        "jav",
        "jpn",
        "kan",
        "kat",
        "kaz",
        "khk",
        "khm",
        "kir",
        "kor",
        "lao",
        "lit",
        "lug",
        "luo",
        "lvs",
        "mai",
        "mal",
        "mar",
        "mkd",
        "mlt",
        "mni",
        "mya",
        "nld",
        "nno",
        "nob",
        "npi",
        "nya",
        "ory",
        "pan",
        "pbt",
        "pes",
        "pol",
        "por",
        "ron",
        "rus",
        "sat",
        "slk",
        "slv",
        "sna",
        "snd",
        "som",
        "spa",
        "srp",
        "swe",
        "swh",
        "tam",
        "tel",
        "tgk",
        "tgl",
        "tha",
        "tur",
        "ukr",
        "urd",
        "uzn",
        "vie",
        "yor",
        "yue",
        "zlm",
        "zul",
    ]

    errors = []
    chrf2 = []
    errors_no_src = []
    chrf2_no_src = []
    print(f"| Language | chrF2++ w/ src_lang | chrF2++ no src_lang |")
    print(f"| :------: | :-----------------: | :-----------------: |")
    for src in language_codes:
        print(f"| {src} |", end="", flush=True)
        triton_score, errors_dict = test_pair(src, "eng")
        print(f" {triton_score:.1f} |", end="", flush=True)
        chrf2.append(triton_score)
        errors.append(errors_dict)

        triton_score, errors_dict = test_pair(src, "eng", use_src=False)
        print(f" {triton_score:.1f} |", flush=True)
        chrf2_no_src.append(triton_score)
        errors_no_src.append(errors_dict)

    mean_score = sum(chrf2) / len(chrf2)
    mean_no_score = sum(chrf2_no_src) / len(chrf2_no_src)
    print(f"| **Mean** | **{mean_score:.2f}** | **{mean_no_score:.2f}** |")

    print(f"Errors when using src_lang")
    for errors_dict in errors:
        pprint(errors_dict)
    print("\n\nErrors when no src_lang")
    for errors_dict in errors_no_src:
        pprint(errors_dict)


if __name__ == "__main__":
    main()
