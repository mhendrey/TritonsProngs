from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import requests
from sklearn import metrics


def predict_lang_ids(
    texts: list[str],
    base_url: str = "http://localhost:8000/v2/models",
    max_workers: int = 60,
):
    n = len(texts)
    predictions = [None] * n
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, text in enumerate(texts):
            inference_json = {
                "inputs": [
                    {
                        "name": "INPUT_TEXT",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": [text],
                    }
                ],
            }
            future = executor.submit(
                requests.post,
                url=f"{base_url}/fasttext_language_identification/infer",
                json=inference_json,
            )
            futures[future] = i

        # Gather results
        for future in as_completed(futures):
            i = futures[future]
            try:
                response_json = future.result().json()
                outputs = response_json["outputs"]
                predictions[i] = outputs[0]["data"][0]
            except:
                return response_json

    return predictions


def main():
    test_langs_f1 = {
        "arb_Arab": 0.969,  # Modern Arabic
        "bam_Latn": 0.613,  # Bambara
        "cat_Latn": 0.993,  # Catalan
        "deu_Latn": 0.991,  # German
        "ell_Grek": 1.000,  # Greek
        "eng_Latn": 0.970,  # English
        "hin_Deva": 0.892,  # Hindi
        "pes_Arab": 0.968,  # Iranian Persian
        "nob_Latn": 0.985,  # Bokmal (Norwegian)
        "pol_Latn": 0.988,  # Polish
        "prs_Arab": 0.544,  # Dari Persian
        "rus_Cyrl": 1.000,  # Russian
        "sin_Sinh": 1.000,  # Sinhala
        "tam_Taml": 1.000,  # Tamil
        "jpn_Jpan": 0.986,  # Japanese
        "kor_Hang": 0.994,  # Korean
        "vie_Latn": 0.991,  # Vietnamese
        "zho_Hans": 0.854,  # Chinese
    }
    # Load dataset
    flores = load_dataset("facebook/flores", "all", split="devtest")
    Y_true = defaultdict(list)
    Y_pred = defaultdict(list)

    for i, batch in enumerate(flores.iter(batch_size=500)):
        print(
            f"Starting on batch {i:03}, batch_size = {len(batch['sentence_eng_Latn'])}"
        )
        for lang in test_langs_f1:
            sent_key = f"sentence_{lang}"
            lang_id, _ = lang.split("_")
            n = len(batch[sent_key])
            Y_true[lang_id] += [lang_id] * n
            Y_pred[lang_id] += predict_lang_ids(batch[sent_key])

    for lang in test_langs_f1:
        lang_id, _ = lang.split("_")
        f1_score = metrics.f1_score(Y_true[lang_id], Y_pred[lang_id], average="micro")
        print(
            f"{lang} N={len(Y_true[lang_id]):} Reported f1={test_langs_f1[lang]:.3f}, Measured={f1_score:.3f}"
        )


if __name__ == "__main__":
    main()
