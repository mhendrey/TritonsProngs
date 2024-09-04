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
  probability for top prediction is below this threshold. Default is 0.95.

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
15 sentences for a given language at one time and submit these to the translate
deployment endpoint that is using SeamlessM4T under the hood. Of course, the translate
BLS deployment is then using the sentencex deployment to split up the text into
sentences again. Similar to the SeamlessM4T, we calculate the chF2++ for each text (now
15 sentences) submitted.

The validation is run over a total of 96 languages, but not exactly sure which language was
added (guessing it was cmn_Hant, which is different from the others and seems added after
the fact). The results for each language are listed below:

### Results
| Language | chrF2++ | chrF2++ (Seamless) |
| :------: | :-----: | :----------------: |
| afr | 67.7 | 75.4 |
| amh | 64.0 | 58.6 |
| arb | 68.6 | 64.7 |
| ary | 59.9 | 52.3 |
| arz | 64.2 | 57.6 |
| asm | 61.2 | 55.8 | 
| azj | 60.0 | 51.6 |
| bel | 59.9 | 51.6 |
| ben | 65.1 | 60.0 |
| bos | 70.7 | 65.9 |
| bul | 70.4 | 65.5 |
| cat | 72.6 | 67.8 |
| ceb | 69.6 | 65.9 |
| ces | 68.8 | 63.5 |
| ckb | 61.5 | 55.9 |
| cmn | 62.4 | 55.5 |
| cmn_Hant | 60.6 | 53.4 |
| cym | 74.7 | 71.1 |
| dan | 72.5 | 68.9 |
| deu | 71.7 | 66.5 |
| ell | 66.3 | 59.6 |
| est | 65.6 | 60.5 |
| eus | 64.5 | 57.8 |
| fin | 63.9 | 58.0 |
| fra | 72.2 | 67.6 |
| fuv | 41.9 | 28.8 |
| gaz | 56.0 | 47.0 |
| gle | 65.5 | 61.0 |
| glg | 70.8 | 65.8 |
| guj | 68.6 | 64.7 |
| heb | 68.7 | 65.3 |
| hin | 67.6 | 62.2 |
| hrv | 67.4 | 61.7 |
| hun | 66.1 | 60.1 |
| hye | 68.2 | 63.0 |
| ibo | 60.3 | 53.1 |
| ind | 68.6 | 65.6 |
| isl | 61.6 | 56.3 |
| ita | 66.3 | 60.0 |
| jav | 66.8 | 61.4 |
| jpn | 54.1 | 45.7 |
| kan | 64.7 | 58.9 |
| kat | 62.3 | 55.3 |
| kaz | 64.4 | 58.2 |
| khk | 60.4 | 52.5 |
| *khm* | 10.0 | 55.4 |
| kir | 58.8 | 50.9 |
| kor | 59.9 | 53.2 |
| lao | 64.9 | 59.0 |
| lit | 63.5 | 56.7 |
| lug | 52.7 | 43.6 |
| luo | 55.6 | 47.4 |
| lvs | 63.9 | 58.9 |
| mai | 69.7 | 65.7 |
| mal | 65.7 | 60.7 |
| mar | 66.9 | 61.9 |
| mkd | 70.9 | 65.8 |
| mlt | 75.4 | 74.1 |
| mni | 58.6 | 50.2 |
| mya | 58.1 | 53.7 |
| nld | 64.3 | 57.8 |
| nno | 70.9 | 66.2 |
| nob | 70.5 | 65.3 |
| npi | 68.2 | 65.1 |
| nya | 58.4 | 50.0 |
| ory | 66.7 | 62.3 |
| pan | 56.4 | 64.2 |
| pbt | 61.6 | 56.9 |
| pes | 66.8 | 61.3 |
| pol | 63.1 | 55.5 |
| por | 74.0 | 69.5 |
| ron | 70.7 | 65.3 |
| rus | 66.6 | 60.1 |
| sat | 41.0 | 28.4 |
| slk | 68.5 | 62.8 |
| slv | 65.2 | 59.3 |
| sna | 58.2 | 50.2 |
| snd | 65.1 | 60.5 |
| som | 57.9 | 50.8 |
| spa | 64.8 | 57.7 |
| srp | 70.9 | 66.6 |
| swe | 72.6 | 69.0 |
| swh | 66.5 | 62.4 |
| tam | 62.9 | 57.3 |
| tel | 67.0 | 62.5 |
| tgk | 63.7 | 58.4 |
| tgl | 69.7 | 65.0 |
| *tha* | 15.6 | 54.5 |
| tur | 66.8 | 60.4 |
| ukr | 67.9 | 62.2 |
| urd | 63.9 | 59.6 |
| uzn | 64.0 | 57.2 |
| vie | 64.5 | 58.9 |
| yor | 51.0 | 41.6 |
| yue | 57.6 | 49.2 |
| zul | 66.5 | 60.6 |
| **Mean** | **63.47** | **58.84** |

Comparing against the single sentence translation, we find that we generally get
slightly better results with an average chrF2++ score of 63.5 compared to the
sentence level comparision of 58.8. The Seamless paper quotes an average of 59.2.
It's worth noting that a few of the results were significantly worse (tha and khm).
These are a result of the
[sentencex](./sentencex.md) failing to split the text into sentences due to these
languages lacking any punctuation. As a result, though Seamless has a context window
large enough to process all the text it generates a stop token after the first
sentence or two causing the scores to crater.

### Code
The code can be found in the [validate.py](../model_repository/translate/validate.py)
file.
