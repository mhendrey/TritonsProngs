#  SeamlessM4T_Text2Text
This deployment hosts the text to text portion of
[SeamlessM4Tv2Large](https://huggingface.co/facebook/seamless-m4t-v2-large)
to perform machine translation. It takes as input the text to be translated along with
the source language identifier (ISO 639-3) and the target language identifier
(ISO 639-3) and returns the translated text. If you don't know the source language,
you can use the
[fastText Language Identifier](./fasttext_language_identification.md)

Like most machine translation models, this model appears to have been trained on
sentence level data. This means that if you provide a long section of text it will
likely stop generating after at most a few sentences. This is because it generates
the `eos_token`. Thus, the text should be split into sentence sized text for
translation. You can use the [sentencex](sentencex.md) deployment to do this for you.
In the future, additional sentence segmenters may be added.

This is a lower level of abstraction, most clients likely should be using
[translate](translate.md) deployment.

## Example Request
Here's an example request. Just a few things to point out
1. "shape": [1, 512] because we have dynamic batching and the first axis is
   the batch size and the second axis is the maximum token size (pad with '1').
2. "data": this should be "row" flattened. It will be reshaped by the server. Also,
   numpy is not serializable, so convert to python list.

```
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"
text = (
    "The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape."
)
src_lang = "eng"
tgt_lang = "fra"

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
        }
    ]
}
model_response = requests.post(
    url=f"{base_url}/seamlessm4t_text2text/infer",
    json=inference_request,
).json()

"""
JSON response output looks like
{
    "model_name": "seamlessm4t_text2text",
    "model_version": "1",
    "outputs": [
        {
            "name": "TRANSLATED_TEXT",
            "datatype": "BYTES",
            "shape": [1, 1],
            "data": [
                "Le caméléon iridescent se promenait à travers le paysage urbain cyberpunk éclairé au néon."
            ]
        }
    ]
}
"""
```

### Sending Many Requests
If you want to send a lot of text for translation, it's important that you send each
request in a multithreaded way to achieve optimal throughout. The example below shows
8 sentences (from two different languages) to be translated.

NOTE: You will encounter a "OSError Too many open files" if you send a lot of requests.
Typically the default ulimit is 1024 on most system. Either increace this using 
`ulimit -n {n_files}`, or don't create too many futures before you process them when
completed.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

base_url = "http://localhost:8000/v2/models"

sentences = [
    'In a groundbreaking development, scientists at MIT have unveiled a new artificial intelligence system capable of predicting earthquakes with unprecedented accuracy.',
    "The AI, dubbed 'TerraWatch,' analyzes seismic data and geological patterns to forecast potential tremors up to two weeks in advance.",
    'Early tests of TerraWatch have shown a 90% success rate in predicting earthquakes of magnitude 5.0 or greater.',
    'This technological leap could revolutionize disaster preparedness and potentially save thousands of lives in earthquake-prone regions around the world.',
    '昨天，实力不济的新西兰国家足球队在世界杯四分之一决赛中以 2-1 的惊人比分战胜巴西队，震惊了体育界。',
    '这场比赛在卡塔尔的卢萨尔体育场举行，新西兰队克服了上半场的劣势，获得了世界杯历史上第一个半决赛席位。',
    '新西兰队队长萨姆·科尔在第 89 分钟打入制胜一球，让球队规模虽小但热情的球迷陷入疯狂。',
    '这场胜利标志着新西兰足球的历史性时刻，并将在半决赛中与常年强队德国队展开一场大卫与歌利亚的较量。'
]
src_langs = ['eng', 'eng', 'eng', 'eng', 'zho', 'zho', 'zho', 'zho']
tgt_langs = ['fra', 'fra', 'fra', 'fra', 'eng', 'eng', 'eng', 'eng']

futures = {}
results = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, (sentence, src_lang, tgt_lang) in enumerate(zip(sentences, src_langs, tgt_langs)):
        inference_request = {
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [sentence],
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
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/seamlessm4t_text2text/infer",
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
                translated_text = response.json()["outputs"][0]["data"][0]
                results[futures[future]] = translated_text
            except Exception as exc:
                raise ValueError(f"Error getting data from response: {exc}")

results_sorted = {k:v for k, v in sorted(results.items())}
for k, v in sorted(results.items()):
    print(f"{k}: {v}")
```

## Performance Analysis
There is some data in [data/seamlessm4t_text2text](../data/seamlessm4t_text2text/load_sample.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container.

```
sdk-container:/workspace perf_analyzer \
    -m seamlessm4t_text2text \
    -v \
    --input-data data/seamlessm4t_text2text/load_sample.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 41.7069 infer/sec. Avg latency: 1407924 usec (std 66717 usec). 
  * Pass [2] throughput: 43.5366 infer/sec. Avg latency: 1339310 usec (std 109078 usec). 
  * Pass [3] throughput: 47.0795 infer/sec. Avg latency: 1309681 usec (std 61867 usec). 
  * Pass [4] throughput: 44.6202 infer/sec. Avg latency: 1336828 usec (std 93723 usec). 
  * Client: 
    * Request count: 3246
    * Throughput: 45.0787 infer/sec
    * Avg client overhead: 0.00%
    * Avg latency: 1328177 usec (standard deviation 50548 usec)
    * p50 latency: 1080924 usec
    * p90 latency: 2106890 usec
    * p95 latency: 2131453 usec
    * p99 latency: 2188437 usec
    * Avg HTTP time: 1328170 usec (send 34 usec + response wait 1328136 usec + receive 0 usec)
  * Server: 
    * Inference count: 3246
    * Execution count: 74
    * Successful request count: 3246
    * Avg request latency: 1327968 usec (overhead 50 usec + queue 330172 usec + compute input 241 usec + compute infer 997110 usec + compute output 395 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 45.0787 infer/sec, latency 1328177 usec



## Validation
To validate that the model is performing as expected, we calculate the performance on the
[BUCC Bitext Mining dataset](https://huggingface.co/datasets/mteb/bucc-bitext-mining)
and compare the performance against results published in the
[Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/abs/2402.05672).

This dataset consists of 4 separate dataset with pairs of sentences in [zh-en, fr-en,
de-en, and ru-en]. Table 5 in the paper reports that the Multilingual E5 large model
achieved **98.6** on this benchmark. Unfortunately the paper doesn't give any details
as to how they did the evaluation. In particular, the BUCC Biitext Mining dataset is
supposed to consist of non-parallel sentences with only about 2-3% of the sentences
having a corresponding translated sentence in the other language. However, the
Huggingface test data has aligned sentences. This may make the task much too easy, but
we will proceed in the absence of more information.

For each language pair dataset, we query with one side and calculate the top-1 accuracy
of finding the corresponding pair in the other language. We calculate a weighted
average across the four sets of language pairs to get a single number. We use
approximate nearest neighbors to perform the search of the 4 nearest neighbors based
upon the cosine distance. We then perform two separate reranking methods before
choosing the top nearest neighbor from this candidate list.  The first is just the
cosine distance itself. The second is based upon a margin scoring approach that is
referenced in the technical approach. This approach is outlined in
[Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings](https://arxiv.org/abs/1811.01136).

The code can be found in the [validate.py](../model_repository/multilingual_e5_large/validate.py)
file.

### Results

| Language Pairs | Margin Accuracy | Cosine Accuracy | # of Records |
| :------------: | :-------------: | :-------------: | :----------: |
| zh-en | 99.53 | 99.26 | 1,899 |
| fr-en | 99.00 | 98.62 | 9,086 |
| de-en | 99.61 | 99.52 | 9,580 |
| ru-en | 97.94 | 97.74 | 14,435|
| **Mean** | **98.76** | **98.54** | |

These match well with the reported 98.6 in the technical report.

### Code
The code is available in [model_repository/multilingual_e5_large/validate.py](../model_repository/multilingual_e5_large/validate.py)