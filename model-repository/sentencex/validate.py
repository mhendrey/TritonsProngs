import json
from pathlib import Path
from pprint import pprint
import requests


def get_sentences(
    text: str, lang_id: str, base_url: str = "http://localhost:8000/v2/models"
):
    inference_json = {
        "inputs": [
            {
                "name": "INPUT_TEXT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [text],
            },
            {
                "name": "LANG_ID",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [lang_id],
            },
        ]
    }
    response_json = requests.post(
        url=f"{base_url}/sentencex/infer", json=inference_json
    ).json()
    try:
        sentences = response_json["outputs"][0]["data"]
    except:
        raise ValueError(f"{response_json}")

    return sentences


def main():
    golden_rules_en = (
        Path.home() / "data" / "golden_rules_sentence_segmenter/golden_rules_en.json"
    )
    data = json.load(golden_rules_en.open())

    n_correct = 0
    for i, record in enumerate(data["en"]):
        sentences = get_sentences(record["text"], "en")
        if len(sentences) != len(record["target"]):
            print(f"{i} Failed by not matching size of array")
            pprint(record)
            pprint(sentences)
            print("")
            continue
        all_match = True
        for s, t in zip(sentences, record["target"]):
            if s != t:
                all_match = False
        if all_match:
            n_correct += 1
        else:
            print(f"{i} Failed")
            print(record)
            print(sentences)
            print("")

    print(f"Fraction Correct: {n_correct / len(data['en']):.4f}")


if __name__ == "__main__":
    main()
