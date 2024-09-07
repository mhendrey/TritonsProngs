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
  * Pass [1] throughput: 39.2406 infer/sec. Avg latency: 1483547 usec (std 95250 usec). 
  * Pass [2] throughput: 41.4547 infer/sec. Avg latency: 1458850 usec (std 112853 usec). 
  * Pass [3] throughput: 40.1635 infer/sec. Avg latency: 1469074 usec (std 108947 usec). 
  * Client: 
    * Request count: 2902
    * Throughput: 40.2859 infer/sec
    * Avg client overhead: 0.00%
    * Avg latency: 1470271 usec (standard deviation 70666 usec)
    * p50 latency: 1521789 usec
    * p90 latency: 1663912 usec
    * p95 latency: 1674379 usec
    * p99 latency: 1712540 usec
    * Avg HTTP time: 1470265 usec (send 239 usec + response wait 1470026 usec + receive 0 usec)
  * Server: 
    * Inference count: 2902
    * Execution count: 92
    * Successful request count: 2902
    * Avg request latency: 1469289 usec (overhead 30 usec + queue 681745 usec + compute input 209 usec + compute infer 787045 usec + compute output 259 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 40.2859 infer/sec, latency 1470271 usec


## Validation
To validate this implementation of SeamlessM4Tv2LargeForTextToText, we use the
same approach outlined in the
[Seamless paper](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/).
In particular we are looking to match the results in Table 7 that calculates the chrF
score on the [Flores-200](https://huggingface.co/datasets/facebook/flores) dataset for
the translated 95 different languages into English (X-eng). The table provides the
average chrF2++ score over those 95 languages of 59.2.

The validation is run over a total of 96 languages, but not exactly sure which language was
added (guessing it was cmn_Hant, which is different from the others and seems added after
the fact). The results for each language are listed below:

### Results

| Language | chrF2++ |
| :------: | :-----: |
| afr | 75.4 |
| amh | 58.6 |
| arb | 64.7 |
| ary | 52.3 |
| arz | 57.6 |
| asm | 55.8 |
| azj | 51.6 |
| bel | 51.6 |
| ben | 60.0 |
| bos | 65.9 |
| bul | 65.5 |
| cat | 67.8 |
| ceb | 65.9 |
| ces | 63.5 |
| ckb | 55.9 |
| cmn | 55.5 |
| cmn_Hant | 53.4 |
| cym | 71.1 |
| dan | 68.9 |
| deu | 66.5 |
| ell | 59.6 |
| est | 60.5 |
| eus | 57.8 |
| fin | 58.0 |
| fra | 67.6 |
| fuv | 28.8 |
| gaz | 47.0 |
| gle | 61.0 |
| glg | 65.8 |
| guj | 64.7 |
| heb | 65.3 |
| hin | 62.2 |
| hrv | 61.7 |
| hun | 60.1 |
| hye | 63.0 |
| ibo | 53.1 |
| ind | 65.6 |
| isl | 56.3 |
| ita | 60.0 |
| jav | 61.4 |
| jpn | 45.7 |
| kan | 58.9 |
| kat | 55.3 |
| kaz | 58.2 |
| khk | 52.5 |
| khm | 55.4 |
| kir | 50.9 |
| kor | 53.2 |
| lao | 59.0 |
| lit | 56.7 |
| lug | 43.6 |
| luo | 47.4 |
| lvs | 58.9 |
| mai | 65.7 |
| mal | 60.7 |
| mar | 61.9 |
| mkd | 65.8 |
| mlt | 74.1 |
| mni | 50.2 |
| mya | 53.7 |
| nld | 57.8 |
| nno | 66.2 |
| nob | 65.3 |
| npi | 65.1 |
| nya | 50.0 |
| ory | 62.3 |
| pan | 64.2 |
| pbt | 56.9 |
| pes | 61.3 |
| pol | 55.5 |
| por | 69.5 |
| ron | 65.3 |
| rus | 60.1 |
| sat | 28.4 |
| slk | 62.8 |
| slv | 59.3 |
| sna | 50.2 |
| snd | 60.5 |
| som | 50.8 |
| spa | 57.7 |
| srp | 66.6 |
| swe | 69.0 |
| swh | 62.4 |
| tam | 57.3 |
| tel | 62.5 |
| tgk | 58.4 |
| tgl | 65.0 |
| tha | 54.5 |
| tur | 60.4 |
| ukr | 62.2 |
| urd | 59.6 |
| uzn | 57.2 |
| vie | 58.9 |
| yor | 41.6 |
| yue | 49.2 |
| zul | 60.6 |
| **Mean** | **58.84**

We find very close agreement with the 59.2 listed in the Seamless paper. Differences are likely
attributed to slight difference in `generation()` arguments. For example, we use a `num_beams=3`
to help limit VRAM needed.

### Code
The code can be found in the [validate.py](../model_repository/seamlessm4t_text2text/validate.py)
file.
