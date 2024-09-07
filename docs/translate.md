# Translate
This is a BLS deployment for clients to sent text to be translated. This BLS is
composed of three subtasks, each of which is it's own deployment:

1. Language Identification
   Identify language of the source text sent if not provided by the client. Default
   model is the [fastText Language Identification model](./fasttext_language_identification.md).
   Other options that may be added in the future include [Lingua](https://github.com/pemistahl/lingua-py)
2. Sentence Segmenter
   Nearly all translation models were trained on sentence level text and thus input
   text needs to be broken up into sentence chunks. Default segmenter is the
   [sentencex](./sentencex.md). Other options that may be added in the future include
   [PySBD](https://github.com/nipunsadvilkar/pySBD).
3. Translation
   Currently using [SeamlessM4Tv2Large](./seamlessm4t_text2text.md) as the default
   translation model given its wide coverage of languages. In the next release,
   [NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M) will be added given
   its faster speed with same or even slightly higher average performance across many
   languages.

General workflow organized by the BLS. If the `src_lang` is provided by the client,
then any language detection step is skipped and sentence segmentation is performed
followed by translation of each of the sentences. The translated results are bundled
together using a simple `" ".join(translated_sentences_array)` and then sent back.

If the `src_lang` is not provided by the client, the entire text provided is sent to
the language identification model to provide the necessary `src_lang` for the sentence
segmentation step. If the probability associated with the top result from the language
identifcation model is below `language_id_threshold`, then the language identification
is run again on each sentence after segmentation before translation occurs. The
translated results are bundled together as above and sent back to the client.

Because dynamic batching has been enabled for these Triton Inference Server
deployments, clients simply send each request separately. This simplifies the code for
the client, see examples below, yet they reap the benefits of batched processing. In
addition, this allows for controlling the GPU RAM consumed by the server.

## Optional Request Parameters
* `src_lang`: ISO 639-3 Language Code for submitted text. Default is `None` which
  triggers using language identification model.
* `tgt_lang`: ISO 639-3 Language Code for translated text. Default is `eng`
* `language_id_threshold`: Run language id for each sentence if document level language
  probability for top prediction is below this threshold. Default is 0.30.

## Send Single Request
```
import requests

base_url = "http://localhost:8000/v2/models"

text = (
    """Dans les ruelles sombres de Neo-Paris, l'année 2077 étale son ombre numérique sur les derniers vestiges d'une humanité en déclin. La ville, désormais contrôlée par des corporations omnipotentes, brille de mille lumières artificielles, cachant la misère de ceux qui errent dans ses interstices numériques. Au cœur de ce chaos urbain, un hacker solitaire, connu sous le pseudonyme de Phoenix, se faufile à travers les réseaux informatiques, laissant sa marque dans le vaste univers virtuel qui enveloppe la réalité. Avec ses yeux augmentés par la cybernétique, il perçoit le monde tel un flux de données, dévoilant les secrets que les puissants cherchent à garder enfouis."""
)
inference_json = {
    "parameters": {"src_lang": "fra"}, # Optional src_lang provided
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [text],
        }
    ]
}
translated_response = requests.post(
    url=f"{base_url}/translate/infer",
    json=inference_json,
)

response_json = translated_response.json()
"""
{
    "model_name": "translate",
    "model_version": "1",
    "outputs": [
        {
            "name": "TRANSLATED_TEXT",
            "shape": [1],
            "datatype": "BYTES",
            "data": [
                'In the dark alleys of Neo-Paris, the year 2077 spreads its digital shadow over the last remnants of a declining humanity. The city, now controlled by omnipotent corporations, shines with a thousand artificial lights, hiding the misery of those who wander in its digital interstices. At the heart of this urban chaos, a lone hacker, known by the pseudonym Phoenix, sneaks through computer networks, leaving his mark in the vast virtual universe that envelops reality. With his cybernetically enhanced eyes, he perceives the world as a flow of data, revealing the secrets that the powerful seek to keep hidden.'
            ]
        }
    ]
}
"""
```

### Sending Many Requests
To submit multiple requests, use multithreading to send the requests in parallel to
take advantage of the dynamic batching on the server end to maximize throughput.

NOTE: You will encounter an "OSError: Too many open files" if you send a lot of
requests. Typically the default ulimit is 1024 on most system. Either increase this
using `ulimit -n {n_files}`, or don't create too many futures before you processing
some of them.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

base_url = "http://localhost:8000/v2/models"

# First is in French, second is in Spanish
texts = [
    """Dans les ruelles sombres de Neo-Paris, l'année 2077 étale son ombre numérique sur les derniers vestiges d'une humanité en déclin. La ville, désormais contrôlée par des corporations omnipotentes, brille de mille lumières artificielles, cachant la misère de ceux qui errent dans ses interstices numériques. Au cœur de ce chaos urbain, un hacker solitaire, connu sous le pseudonyme de Phoenix, se faufile à travers les réseaux informatiques, laissant sa marque dans le vaste univers virtuel qui enveloppe la réalité. Avec ses yeux augmentés par la cybernétique, il perçoit le monde tel un flux de données, dévoilant les secrets que les puissants cherchent à garder enfouis.""",
    """Las luces de neón arrojaban arcoíris digitales a través de los callejones goteantes de Neo-París, una sinfonía caótica de luces y sombras donde el acero tosco se entrelazaba con hologramas relucientes. El viento, saturado de vapores químicos y sueños erróneos, silbaba entre los rascacielos, llevando consigo el murmullo de una ciudad donde los humanos se disolvían en la matriz, buscando un escape hacia los algoritmos y las sombras digitales. Fue en este océano de datos y desilusión donde yo, Kaï, un cazador de fallas a sueldo de un jefe enigmático, me lancé hacia una misión que sacudiría los cimientos mismos de nuestra fracturada realidad.""",
]

futures = {}
translated = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, text in enumerate(texts):
        infer_request = {
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
            url=f"{base_url}/translate/infer",
            json=infer_request,
        )
        futures[future] = i
    
    for future in as_completed(futures):
        try:
            response = future.result()
        except Exception as exc:
            print(f"{futures[future]} threw {exc}")
        try:
            translated_text = response.json()["outputs"][0]["data"]
        except Exception as exc:
            raise ValueError(f"Error getting data from response: {exc}")
        translated[futures[future]] = translated_text
print(translated)
```

### Performance Analysis
There is some data in [data/translate](../data/translate/load_sample_one.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container to measure the throughput. This data contains a single spanish news
article with 21 sentences.

```
sdk-container:/workspace perf_analyzer \
    -m translate \
    -v \
    --input-data data/translate/load_sample_one.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --max-trials=4 \
    --concurrency-range=3 \
    --bls-composing=fasttext_language_identification,sentencex,seamlessm4t_text2text
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 3
  * Pass [1] throughput: 1.4166 infer/sec. Avg latency: 2063479 usec (std 445241 usec). 
  * Pass [2] throughput: 1.41662 infer/sec. Avg latency: 2064923 usec (std 411890 usec). 
  * Pass [3] throughput: 1.37496 infer/sec. Avg latency: 2138226 usec (std 537251 usec). 
  * Client: 
    * Request count: 101
    * Throughput: 1.40272 infer/sec
    * Avg client overhead: 0.00%
    * Avg latency: 2088387 usec (standard deviation 174699 usec)
    * p50 latency: 1965212 usec
    * p90 latency: 2791654 usec
    * p95 latency: 2804253 usec
    * p99 latency: 3171326 usec
    * Avg HTTP time: 2088377 usec (send 52 usec + response wait 2088325 usec + receive 0 usec)
  * Server: 
    * Inference count: 101
    * Execution count: 100
    * Successful request count: 101
    * Avg request latency: 2088165 usec (overhead 223773 usec + queue 916062 usec + compute 948330 usec)

  * Composing models: 
  * fasttext_language_identification, version: 1
      * Inference count: 104
      * Execution count: 100
      * Successful request count: 104
      * Avg request latency: 1644 usec (overhead 2 usec + queue 339 usec + compute input 13 usec + compute infer 1277 usec + compute output 11 usec)

  * seamlessm4t_text2text, version: 1
      * Inference count: 2149
      * Execution count: 81
      * Successful request count: 2149
      * Avg request latency: 1859923 usec (overhead 28 usec + queue 915307 usec + compute input 188 usec + compute infer 944142 usec + compute output 257 usec)

  * sentencex, version: 1
      * Inference count: 104
      * Execution count: 98
      * Successful request count: 104
      * Avg request latency: 2858 usec (overhead 3 usec + queue 416 usec + compute input 14 usec + compute infer 2413 usec + compute output 10 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 3, throughput: 1.40272 infer/sec, latency 2088387 usec

### Validation
We use the same [Flores dataset](https://huggingface.co/datasets/facebook/flores)
used to validate the
[SeamlessM4Tv2ForTextToText](seamlessm4t_text2text.md), but this time, we aggregate up
15 sentences for a given language at one time and submit these to the `translate`
deployment endpoint that is using SeamlessM4T under the hood. Of course, the
`translate` deployment is using the `sentencex` deployment to split the text up into
sentences again. However, the chF2++ metric uses the block of 15 sentences for
comparison. For each language, we perform translation first by providing the
`src_lang` as a request parameter. This causes `translate` to skip language detection.
We then repeat doing the translation, but without providing the `src_lang`. This
causes `translate` to use the language detection deployment before performing sentence
segmentation followed by translation. In addition, if the probability assigned to the
top predicted language is less than the `language_id_threhold` (0.30), then each
sentence in the segmenter is sent for language detection before being translated.

The validation is run over a total of 96 languages. The results for each language are
listed below:

### Results
| Language | chrF2++ w/ src_lang | chrF2++ no src_lang |
| :------: | :-----------------: | :-----------------: |
| afr | 67.7 | 67.7 | 
| amh | 64.0 | 64.0 | 
| arb | 68.6 | 68.6 | 
| ary | 59.9 | 58.6 | 
| arz | 64.1 | 63.2 | 
| asm | 61.2 | 61.2 | 
| azj | 60.0 | 60.0 | 
| bel | 59.9 | 59.9 | 
| ben | 65.1 | 65.2 | 
| bos | 70.7 | 70.7 | 
| bul | 70.4 | 70.4 | 
| cat | 72.6 | 72.6 | 
| ceb | 69.7 | 69.6 | 
| ces | 68.8 | 68.8 |
| ckb | 61.5 | 61.5 |
| cmn | 62.4 | 61.8 |
| cmn_Hant | 60.6 | 55.9 |
| cym | 74.7 | 74.7 |
| dan | 72.5 | 72.6 |
| deu | 71.7 | 71.7 |
| ell | 66.3 | 66.3 |
| est | 65.6 | 65.6 |
| eus | 64.5 | 64.5 |
| fin | 63.9 | 63.9 |
| fra | 72.2 | 72.2 |
| fuv | 41.9 | 41.9 |
| gaz | 56.0 | 56.0 |
| gle | 65.5 | 65.5 |
| glg | 70.8 | 70.8 |
| guj | 68.6 | 68.6 |
| heb | 68.7 | 68.8 |
| hin | 67.6 | 67.6 |
| hrv | 67.4 | 67.3 |
| hun | 66.1 | 66.1 |
| hye | 68.2 | 68.2 |
| ibo | 60.3 | 60.3 |
| ind | 68.7 | 68.6 |
| isl | 61.6 | 61.6 |
| ita | 66.3 | 66.3 |
| jav | 66.8 | 66.8 |
| jpn | 54.1 | 54.1 |
| kan | 64.7 | 64.8 |
| kat | 62.3 | 62.3 |
| kaz | 64.4 | 64.4 |
| khk | 60.3 | 60.3 |
| **khm** | 10.0 | 10.0 |
| kir | 58.8 | 58.8 |
| kor | 59.9 | 59.9 |
| lao | 64.9 | 64.9 |
| lit | 63.5 | 63.5 |
| lug | 52.7 | 52.7 |
| luo | 55.6 | 55.6 |
| lvs | 64.0 | 64.0 |
| mai | 69.7 | 69.7 |
| mal | 65.8 | 65.7 |
| mar | 66.9 | 66.9 |
| mkd | 70.9 | 70.9 |
| mlt | 75.4 | 75.4 |
| mni | 58.6 | 58.6 |
| mya | 58.1 | 58.1 |
| nld | 64.3 | 64.3 |
| nno | 70.9 | 70.9 |
| nob | 70.5 | 70.5 |
| npi | 68.3 | 68.3 |
| nya | 58.4 | 58.4 |
| ory | 66.6 | 66.7 |
| pan | 56.4 | 56.4 |
| pbt | 61.6 | 61.6 |
| pes | 66.8 | 66.7 |
| pol | 63.1 | 63.1 |
| por | 74.0 | 74.0 |
| ron | 70.7 | 70.7 |
| rus | 66.7 | 66.6 |
| sat | 41.0 | 41.0 |
| slk | 68.5 | 68.5 |
| slv | 65.2 | 65.2 |
| sna | 58.2 | 58.2 |
| snd | 65.1 | 65.1 |
| som | 57.9 | 57.9 |
| spa | 64.8 | 64.8 |
| srp | 70.9 | 70.9 |
| swe | 72.6 | 72.6 |
| swh | 66.4 | 66.4 |
| tam | 62.9 | 62.9 |
| tel | 67.0 | 67.0 |
| tgk | 63.7 | 63.7 |
| tgl | 69.6 | 69.6 |
| **tha** | 15.4 | 15.5 |
| tur | 66.8 | 66.8 |
| ukr | 67.9 | 67.9 |
| urd | 63.9 | 63.9 |
| uzn | 64.0 | 64.0 |
| vie | 64.5 | 64.5 |
| yor | 51.0 | 51.0 |
| yue | 57.6 | 57.6 |
| zul | 66.5 | 66.5 |
| **Mean** | **63.47** | **63.39** |


Comparing against the single sentence translation, we find that we generally get
slightly better results with an average chrF2++ score of 63.5 compared to the
sentence level comparision of 58.8. The Seamless paper quotes an average of 59.2.
It's worth noting that a few of the results were significantly worse (tha and khm).
These are a result of the
[sentencex](./sentencex.md) failing to split the text into sentences due to these
languages lacking any punctuation. As a result, though Seamless has a context window
large enough to process all the text it generates a stop token after the first
sentence or two causing the scores to crater.

In addition, the cmn_Hant also struggles a little bit. This is due to the language
detection struggling to identify the correct language for this particular language.

### Code
The code can be found in the [validate.py](../model_repository/translate/validate.py)
file.
