from collections import defaultdict
from datasets import load_dataset
import requests
from sklearn import metrics


def predict_lang_id(
    text: str,
    top_k: int = 1,
    threshold: float = 0.0,
    base_url: str = "http://localhost:8000/v2/models",
):
    inference_json = {
        "parameters": {"top_k": top_k, "threshold": threshold},
        "inputs": [
            {
                "name": "INPUT_TEXT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [text],
            }
        ],
    }
    response_json = requests.post(
        url=f"{base_url}/fasttext_language_identification/infer",
        json=inference_json,
    ).json()
    try:
        outputs = response_json["outputs"]
        result = []
        for lang_id, script, prob in zip(
            outputs[0]["data"], outputs[1]["data"], outputs[2]["data"]
        ):
            result.append({"lang_id": lang_id, "script": script, "probability": prob})
        return result
    except:
        return response_json


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
    for batch in flores.iter(batch_size=500):
        for lang in test_langs_f1:
            sent_key = f"sentence_{lang}"
            for sentence in batch[sent_key]:
                lang_id_true, _ = lang.split("_")
                Y_true[lang_id_true].append(lang_id_true)
                Y_pred[lang_id_true].append(predict_lang_id(sentence)[0]["lang_id"])

    for lang in test_langs_f1:
        lang_id, _ = lang.split("_")
        f1_score = metrics.f1_score(Y_true[lang_id], Y_pred[lang_id], average="micro")
        print(f"{lang} Reported f1={test_langs_f1[lang]:.3f}, Measured={f1_score:.3f}")


if __name__ == "__main__":
    main()
