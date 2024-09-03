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
  * Pass [1] throughput: 1.33328 infer/sec. Avg latency: 2149660 usec (std 519525 usec). 
  * Pass [2] throughput: 1.37495 infer/sec. Avg latency: 2126508 usec (std 426229 usec). 
  * Pass [3] throughput: 1.45827 infer/sec. Avg latency: 2144916 usec (std 435295 usec). 
  * Client: 
    * Request count: 100
    * Throughput: 1.38884 infer/sec
    * Avg client overhead: 0.00%
    * Avg latency: 2140359 usec (standard deviation 148598 usec)
    * p50 latency: 1995987 usec
    * p90 latency: 2859377 usec
    * p95 latency: 2875004 usec
    * p99 latency: 3307600 usec
    * Avg HTTP time: 2140350 usec (send 48 usec + response wait 2140302 usec + receive 0 usec)
  * Server: 
    * Inference count: 100
    * Execution count: 99
    * Successful request count: 100
    * Avg request latency: 2140152 usec (overhead 194297 usec + queue 967685 usec + compute 978170 usec)

  * Composing models: 
  * fasttext_language_identification, version: 1
      * Inference count: 103
      * Execution count: 100
      * Successful request count: 103
      * Avg request latency: 2162 usec (overhead 3 usec + queue 467 usec + compute input 15 usec + compute infer 1661 usec + compute output 15 usec)

  * seamlessm4t_text2text, version: 1
      * Inference count: 2100
      * Execution count: 77
      * Successful request count: 2100
      * Avg request latency: 1940889 usec (overhead 28 usec + queue 966532 usec + compute input 195 usec + compute infer 973893 usec + compute output 241 usec)

  * sentencex, version: 1
      * Inference count: 103
      * Execution count: 102
      * Successful request count: 103
      * Avg request latency: 2837 usec (overhead 2 usec + queue 686 usec + compute input 12 usec + compute infer 2127 usec + compute output 9 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 3, throughput: 1.38884 infer/sec, latency 2140359 usec

