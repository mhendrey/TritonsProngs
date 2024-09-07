#  fastText Language Identification
This deployment hosts the [fastText Language Identification](https://huggingface.co/facebook/fasttext-language-identification).
Given some input text, it predicts the language, script, and probability. This is
especially useful for machine translation since nearly all translation models require
you to provide the source language.

This is a very lightweight model that runs on the CPU. Dynamic batching is enabled
(though doesn't save much except some networking overhead), so clients simply send in
each request separately.

The model sends back three arrays
  * SRC_LANG: List of predicted language codes sorted most likely to least. Seems to
    mostly be the ISO 639-3 codes according to the [NLLB paper](https://research.facebook.com/file/585831413174038/No-Language-Left-Behind--Scaling-Human-Centered-Machine-Translation.pdf),
    but not always. See for example, "arb" for Arabic, but the ISO 639-3 for Arabic is
    "ara".
  * SRC_SCRIPT: List of the accompanying script
  * PROBABILITY: List of the accompanying probability

By default, the model will return just the most likely answer. You may return more than
the most likely answer, but sending in an optional request parameter `top_k` greater
than the default of 1. In addition, the optional request parameter `threshold` will
only return predicted languages/scripts whose probability exceeds the `threshold`. The
default value is 0.0.

| Request Parameter | Type | Default Value | Description |
| :---------------: | :--: | :-----------: | :---------: |
| top_k | int | 1 | Number of top predicted languages to return |
| threshold | float | 0.0 | Only return predicted language if probability exceeds this value |


## Example Request
Here's an example request. Just a few things to point out
1. "shape": [1, 1] because we have dynamic batching and the first axis is
   the batch size and the second axis means we send just one text string.
2. "datatypes": This is "BYTES", but you can send a string. It will be utf-8 converted

```
import requests

base_url = "http://localhost:8000/v2/models"
text = (
    "The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape."
)

inference_request = {
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [text],
        }
    ]
}
model_response = requests.post(
    url=f"{base_url}/fasttext_language_identification/infer",
    json=inference_request,
).json()

"""
JSON response output looks like
{
    'model_name': 'fasttext_language_identification',
    'model_version': '1',
    'outputs': [
        {
            'name': 'SRC_LANG',
            'datatype': 'BYTES',
            'shape': [1, 1],
            'data': ['eng']
        },
        {
            'name': 'SRC_SCRIPT',
            'datatype': 'BYTES',
            'shape': [1, 1],
            'data': ['Latn']
        },
        {
            'name': 'PROBABILITY',
            'datatype': 'FP64',
            'shape': [1, 1],
            'data': [0.9954364895820618]}
    ]
}
"""
```

### Example Sending Optional Request Parameter
We send the same text, but this time set the optional request parameters to return
the top 3 predicted languages, but only if their probability exceeds 0.00122. In this
case, only the first two predictions exceed the threshold. Thus, just two results come
back in the response.

```
import requests

base_url = "http://localhost:8000/v2/models"
text = (
    "The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape."
)

inference_request = {
    "parameters": {"top_k": 3, "threshold": 0.00122},
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [text],
        }
    ]
}
model_response = requests.post(
    url=f"{base_url}/fasttext_language_identification/infer",
    json=inference_request,
).json()

"""
JSON response output looks like
{
    'model_name': 'fasttext_language_identification',
    'model_version': '1',
    'outputs': [
        {
            'name': 'SRC_LANG',
            'datatype': 'BYTES',
            'shape': [1, 2],
            'data': ['eng', 'kor']
        },
        {
            'name': 'SRC_SCRIPT',
            'datatype': 'BYTES',
            'shape': [1, 2],
            'data': ['Latn', 'Hang']
        },
        {
            'name': 'PROBABILITY',
            'datatype': 'FP64',
            'shape': [1, 2],
            'data': [0.9954364895820618, 0.001247989828698337]}
    ]
}
"""
```


### Sending Many Requests
Though this model is very fast, it is still good practice to send many requests in a
multithreaded way to achieve optimal throughput. Here's an example of sending 100
different text strings to the model.

NOTE: You will encounter a "OSError Too many open files" if you send a lot of requests.
Typically the default ulimit is 1024 on most system. Either increace this using 
`ulimit -n {n_files}`, or don't create too many futures before you process them when
completed.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

base_url = "http://localhost:8000/v2/models"
texts = [
    "Amidst the whispering winds, the ancient castle stood, its stone walls echoing tales of forgotten kingdoms.",                # English
    "El río serpentea a través del valle verde.", # Spanish
    "Le ciel est bleu et le soleil brille.",      # French
    "Die Bücher auf dem Tisch gehören mir.",      # German
    " 他喜欢在清晨散步。",                           # Chinese
    "الليلُ غطَى السماءَ بنجومها اللامعة.",           # Arabic
    "Дерево́ опусти́ло свои́ ветви́ к земле́.",        # Russian
    "あのレストランは海鮮料理で有名です。",             # Japanese
    " बादल आकाश में फैल गए।",                         # Hindi
    "Nyoka huyu hana hamu.",                      # Swahili
]

futures = {}
results = [None] * len(texts)
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, text in enumerate(texts):
        inference_request = {
            "parameters": {"top_k": 2, "threshold": 0.05},
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [text],
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/fasttext_language_identification/infer",
            json=inference_request,
        )
        futures[future] = i
    
    for future in as_completed(futures):
        try:
            response = future.result()
        except Exception as exc:
            print(f"{futures[future]} threw {exc}")
        else:
            try:
                src_langs = response.json()["outputs"][0]["data"]
                src_scripts = response.json()["outputs"][1]["data"]
                probs = response.json()["outputs"][2]["data"]
                results[futures[future]] = (src_langs, src_scripts, probs)
            except Exception as exc:
                raise ValueError(f"Error getting data from response: {exc} {response.json()}")

print(results)
```

## Performance Analysis
There is some data in [data/fasttext_language_identification](../data/fasttext_language_identification/input_text.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container.

```
sdk-container:/workspace perf_analyzer \
    -m fasttext_language_identification \
    -v \
    --input-data data/fasttext_language_identification/input_text.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 8936.82 infer/sec. Avg latency: 6708 usec (std 3498 usec). 
  * Pass [2] throughput: 8626.91 infer/sec. Avg latency: 6949 usec (std 3761 usec). 
  * Pass [3] throughput: 8646.36 infer/sec. Avg latency: 6934 usec (std 3793 usec). 
  * Client: 
    * Request count: 641517
    * Throughput: 8734.72 infer/sec
    * Avg client overhead: 0.79%
    * Avg latency: 6864 usec (standard deviation 3688 usec)
    * p50 latency: 5798 usec
    * p90 latency: 11545 usec
    * p95 latency: 14224 usec
    * p99 latency: 20263 usec
    * Avg HTTP time: 6857 usec (send 31 usec + response wait 6826 usec + receive 0 usec)
  * Server: 
    * Inference count: 641517
    * Execution count: 12838
    * Successful request count: 641517
    * Avg request latency: 7036 usec (overhead 139 usec + queue 1635 usec + compute input 305 usec + compute infer 4353 usec + compute output 602 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 8734.72 infer/sec, latency 6864 usec


## Validation
Validation of the model is done using the [Flores-200](https://huggingface.co/datasets/facebook/flores)
dataset. This is the dataset Meta put together for creating this model and the No
Language Left Behind machine translation model. The results of the language
identification model are reported in the [NLLB paper](https://arxiv.org/abs/2207.04672)
in Table 49. For validating this deployment, a few languages were chosen to allow for
comparison. 

The Flores-200 dataset in Huggingface has both "dev" and "devtest" available. The
"devtest" split is used for validation. This provides 1,012 sentences for each of the
204 language + script combinations available. Each record, contains the same sentence
in the different languge + script combinations.  Validation is done using 18 different
{lang_id}_{script}.

Only the top predicted language is used to determine the F1 score in order to compare
to Table 49 in the NLLB paper.  The results are shown here:

### Results

| Language | Num Records | Reported F1 | Measured F1 |
| :------: | :---------: | :---------: | :---------: |
| arb_Arab | 1012 | 0.969 | 1.000 |
| bam_Latn | 1012 | 0.613 | 0.881 |
| cat_Latn | 1012 | 0.993 | 1.000 |
| deu_Latn | 1012 | 0.991 | 1.000 |
| ell_Grek | 1012 | 1.000 | 1.000 |
| eng_Latn | 1012 | 0.970 | 1.000 |
| hin_Deva | 1012 | 0.892 | 0.998 |
| pes_Arab | 1012 | 0.968 | 0.983 |
| nob_Latn | 1012 | 0.985 | 0.993 |
| pol_Latn | 1012 | 0.988 | 1.000 |
| prs_Arab | 1012 | 0.544 | 0.333 |
| rus_Cyrl | 1012 | 1.000 | 1.000 |
| sin_Sinh | 1012 | 1.000 | 1.000 |
| tam_Taml | 1012 | 1.000 | 1.000 |
| jpn_Jpan | 1012 | 0.986 | 0.982 |
| kor_Hang | 1012 | 0.994 | 1.000 |
| vie_Latn | 1012 | 0.991 | 1.000 |
| zho_Hans | 1012 | 0.854 | 0.818 |

For the most part these seem to agree well. In most cases, the reported matches or
exceeds the measured except for Dari (prs) and Chinese (zho). Using the "dev" split,
gives different results for those two languages. This suggests that more data is
needed to get a more stable result to match the paper's published results. But for this
purpose, the model seems to be working as intended.

### Code
The code is available in [model_repository/fasttext_language_identification/validate.py](../model_repository/fasttext_language_identification/validate.py)