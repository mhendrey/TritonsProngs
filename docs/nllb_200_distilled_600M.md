#  NLLB-200 (Distilled 600M)
This deployment hosts the
[nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
for machine translation. It takes as input the text to be translated along with
the source language identifier (ISO 639-3) and the target language identifier
(ISO 639-3) and returns the translated text. If you don't know the source language,
you can use the
[fastText Language Identifier](./fasttext_language_identification.md)

**NOTE** NLLB uses the language id & script, e.g., "eng_Latn" as the src/tgt lang.

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
1. "shape": [1, 1] because we have dynamic batching and the first axis is
   the batch size and the second axis the number of elements to be translated (always 1).
2. "data": this should be "row" flattened. It will be reshaped by the server. Also,
   numpy is not serializable, so convert to python list.

```
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"
text = (
    "The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape."
)
src_lang = "eng_Latn"
tgt_lang = "fra_Latn"

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
    url=f"{base_url}/nllb_200_distilled_600M/infer",
    json=inference_request,
).json()

"""
JSON response output looks like. This is not as good as the Seamless translation
{
    "model_name": "nllb_200_distilled_600M",
    "model_version": "1",
    "outputs": [
        {
            "name": "TRANSLATED_TEXT",
            "datatype": "BYTES",
            "shape": [1, 1],
            "data": [
                "Le chaméléon iridescent a traversé le cyberpunk à neon."
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
src_langs = ['eng_Latn', 'eng_Latn', 'eng_Latn', 'eng_Latn', 'zho_Hans', 'zho_Hans', 'zho_Hans', 'zho_Hans']
tgt_langs = ['fra_Latn', 'fra_Latn', 'fra_Latn', 'fra_Latn', 'eng_Latn', 'eng_Latn', 'eng_Latn', 'eng_Latn']

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
            url=f"{base_url}/nllb_200_distilled_600M/infer",
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
There is some data in [data/nllb_200_distilled_600M](../data/nllb_200_distilled_600M/load_sample.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container.

```
sdk-container:/workspace perf_analyzer \
    -m nllb_200_distilled_600M \
    -v \
    --input-data data/nllb_200_distilled_600M/load_sample.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 160.527 infer/sec. Avg latency: 371408 usec (std 53577 usec). 
  * Pass [2] throughput: 161.954 infer/sec. Avg latency: 369373 usec (std 37835 usec). 
  * Pass [3] throughput: 161.963 infer/sec. Avg latency: 369320 usec (std 37378 usec). 
  * Client: 
    * Request count: 11629
    * Throughput: 161.481 infer/sec
    * Avg client overhead: 0.01%
    * Avg latency: 370030 usec (standard deviation 17635 usec)
    * p50 latency: 430452 usec
    * p90 latency: 460940 usec
    * p95 latency: 468470 usec
    * p99 latency: 487289 usec
    * Avg HTTP time: 370025 usec (send 90 usec + response wait 369935 usec + receive 0 usec)
  * Server: 
    * Inference count: 11629
    * Execution count: 324
    * Successful request count: 11629
    * Avg request latency: 370637 usec (overhead 13 usec + queue 150243 usec + compute input 247 usec + compute infer 220131 usec + compute output 1 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 161.481 infer/sec, latency 370030 usec


## Validation
To validate this implementation of NLLB-200-Distilled-600M, we use the
same approach outlined in the
[Seamless paper](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/).
In particular we are looking to match the results in Table 7 that calculates the chrF
score on the [Flores-200](https://huggingface.co/datasets/facebook/flores) dataset for
the translated 95 different languages into English (X-eng). The table provides the
average chrF2++ score over those 95 languages of 59.2.

The validation is run over a total of 96 languages, but not exactly sure which language was
added. The results for each language are listed below:

### Results

| Language | chrF2++ |
| :------: | :-----: |
| afr_Latn | 71.7 |
| amh_Ethi | 52.2 |
| arb_Arab | 60.4 |
| ary_Arab | 49.3 |
| arz_Arab | 54.5 |
| asm_Beng | 51.0 |
| azj_Latn | 49.4 |
| bel_Cyrl | 48.4 |
| ben_Beng | 56.2 |
| bos_Latn | 61.9 |
| bul_Cyrl | 61.8 |
| cat_Latn | 64.8 |
| ceb_Latn | 61.1 |
| ces_Latn | 60.0 |
| ckb_Arab | 53.1 |
| cym_Latn | 66.2 |
| dan_Latn | 66.2 |
| deu_Latn | 63.3 |
| ell_Grek | 57.2 |
| est_Latn | 56.5 |
| eus_Latn | 54.1 |
| fin_Latn | 54.5 |
| fra_Latn | 64.0 |
| fuv_Latn | 29.8 |
| gaz_Latn | 42.3 |
| gle_Latn | 56.0 |
| glg_Latn | 62.7 |
| guj_Gujr | 61.3 |
| heb_Hebr | 60.2 |
| hin_Deva | 61.0 |
| hrv_Latn | 58.1 |
| hun_Latn | 56.0 |
| hye_Armn | 57.8 |
| ibo_Latn | 47.4 |
| ind_Latn | 63.0 |
| isl_Latn | 50.5 |
| ita_Latn | 58.3 |
| jav_Latn | 56.9 |
| jpn_Jpan | 48.4 |
| kan_Knda | 55.3 |
| kat_Geor | 50.8 |
| kaz_Cyrl | 53.7 |
| khk_Cyrl | 46.7 |
| khm_Khmr | 51.8 |
| kir_Cyrl | 46.7 |
| kor_Hang | 50.8 |
| lao_Laoo | 54.6 |
| lit_Latn | 53.2 |
| lug_Latn | 40.4 |
| luo_Latn | 41.4 |
| lvs_Latn | 54.3 |
| mai_Deva | 61.4 |
| mal_Mlym | 57.4 |
| mar_Deva | 56.8 |
| mkd_Cyrl | 61.7 |
| mlt_Latn | 70.3 |
| mni_Beng | 46.6 |
| mya_Mymr | 49.1 |
| nld_Latn | 55.7 |
| nno_Latn | 62.1 |
| nob_Latn | 60.8 |
| npi_Deva | 60.6 |
| nya_Latn | 45.8 |
| ory_Orya | 56.9 |
| pan_Guru | 60.1 |
| pbt_Arab | 53.1 |
| pes_Arab | 57.1 |
| pol_Latn | 53.1 |
| por_Latn | 67.7 |
| ron_Latn | 64.6 |
| rus_Cyrl | 56.7 |
| slk_Latn | 59.7 |
| slv_Latn | 56.1 |
| sna_Latn | 46.2 |
| snd_Arab | 58.9 |
| som_Latn | 47.5 |
| spa_Latn | 56.2 |
| srp_Cyrl | 62.0 |
| swe_Latn | 65.3 |
| swh_Latn | 59.0 |
| tam_Taml | 54.2 |
| tel_Telu | 59.7 |
| tgk_Cyrl | 53.7 |
| tgl_Latn | 62.2 |
| tha_Thai | 51.1 |
| tur_Latn | 58.2 |
| ukr_Cyrl | 59.2 |
| urd_Arab | 56.2 |
| uzn_Latn | 54.1 |
| vie_Latn | 56.6 |
| yor_Latn | 38.9 |
| yue_Hant | 49.5 |
| zho_Hans | 51.4 |
| zho_Hant | 46.3 |
| zsm_Latn | 63.0 |
| zul_Latn | 55.3 |
| **Mean** | **55.68**

This is a little bit less than the 58.84 obtained with the
[Seamless](./seamlessm4t_text2text.md) validation. This is mostly due to the difference
in generation configuration. For the Seamless model, we utilize `num_beams=3`, but for
NLLB we found that using `num_beams` greater than 1 caused a massive lowering of the
throughput. As a result, we set `num_beams=1`.

### Code
The code can be found in the [validate.py](../model_repository/nllb_200_distilled_600M/validate.py)
file.
