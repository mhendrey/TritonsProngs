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
                url="http://localhost:8000/v2/models/seamlessm4t_text2text/infer",
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
    print(f"| Language | chrF2++ |")
    print(f"| :------: | :-----: |")
    for src in language_codes:
        try:
            triton_score = test_pair(src, "eng")
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
