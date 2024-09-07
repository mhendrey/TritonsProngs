#  Sentencex
This deployment hosts the [Sentencex](https://github.com/wikimedia/sentencex)
deployment. This is a simple Python library for sentence segmentation which is
typically needed for nearly all machine translation models since the are most commonly
trained on parallel sentences.

Given input text and the language code (ISO 639-1 or ISO 639-3 which get converted to
ISO 639-1), it does it's best to break the text into an array of the corresponding
sentences. This is a very lightweight function that runs on the CPU. Dynamic batching
is enabled (because that's just good habit), but provides little additional throughput
since the library itself is really fast and does not support parallel processing.

The model sends back one array
  * SENTENCES: List of the resulting sentence segments of the input text

## Example Request
Here's an example request. Just a few things to point out
1. "shape": [1, 1] because we have dynamic batching and the first axis is
   the batch size and the second axis means we send just one text string.
2. "datatypes": This is "BYTES", but you can send a string. It will be utf-8 converted

```
import requests

base_url = "http://localhost:8000/v2/models"
text = (
    """Text messaging has transformed how we communicate, with phrases like "i.e." and "e.g." commonly used to clarify points, while "a.m." and "p.m." help specify times efficiently. In emails, it's not uncommon to see "etc." at the end of lists or "i.e." before explanations to avoid redundancy. However, when drafting formal documents or academic papers, one should be cautious with the overuse of abbreviations like "cf." for "compare" or "viz." for "namely" to ensure clarity and professionalism."""
)

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
            "data": ["eng"]
        }
    ]
}
model_response = requests.post(
    url=f"{base_url}/sentencex/infer",
    json=inference_request,
).json()

"""
JSON response output looks like
{'model_name': 'sentencex',
 'model_version': '1',
 'outputs': [{
    'name': 'SENTENCES',
    'datatype': 'BYTES',
    'shape': [1, 3],
    'data': [
        'Text messaging has transformed how we communicate, with phrases like "i.e." and "e.g." commonly used to clarify points, while "a.m." and "p.m." help specify times efficiently.',
        'In emails, it\'s not uncommon to see "etc." at the end of lists or "i.e." before explanations to avoid redundancy.',
        'However, when drafting formal documents or academic papers, one should be cautious with the overuse of abbreviations like "cf." for "compare" or "viz." for "namely" to ensure clarity and professionalism.'
    ]
 }]
}
"""
```

### Sending Many Requests
Though this model is very fast, it is still good practice to send many requests in a
multithreaded way to achieve optimal throughput. Here's an example of sending 5
different text strings to the model.

NOTE: You will encounter a "OSError Too many open files" if you send a lot of requests.
Typically the default ulimit is 1024 on most system. Either increase this using
`ulimit -n {n_files}`, or don't create too many futures before you process them when
completed.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

base_url = "http://localhost:8000/v2/models"
lang_ids = ["fr", "spa", "deu", "jpn", "arb",]
texts = [
    """Le vent glacial sifflait à travers les meurtrières de l'ancienne demeure, emportant avec lui le parfum de poussière et de secrets oubliés. Lorsque le corps d'Antoine Dufour fut découvert dans le jardin, une cascade de violettes immaculées gisant à ses côtés, la tranquillité de la campagne bourguignonne céda la place à une inquiétude palpable. Qui aurait osé briser ce silence et à quel prix ?""",
    """La niebla se aferraba al callejón como un sudario, ocultando las fachadas desgastadas de los edificios y amplificando el eco de los pasos vacilantes de la inspectora Flores. Un cuerpo sin rostro, envuelto en un antiguo tapiz flamenco, yacía en el centro, desafiando el silencio con la mirada vacía de sus ojos de cristal. ¿Qué historia susurraba ese tejido ancestral, y quién se escondía detrás de la máscara de la muerte?""",
    """Ein kalter Schauer kroch entlang des Grauwacke-Pflasterwegs, als Kommissar Becker die geschlossene Villa betrat. Der Duft von verrottenden Rosen und einem undefinierbaren metallischen Geschmack hing schwer in der Luft. Auf dem Boden, inmitten zerbrechlicher Porzellankunst, lag ein Mann, sein Blick fixiert auf ein verschwundenes Detail, das nur er zu sehen schien - ein Rätsel, das nun Beckers Aufgabe war.""",
    """薄暗がりの中、古びた書院のフローリングで、静寂が支配していた。刀の鍔が光る影、そこに倒れた男の姿。彼の白い着物に染み付いた朱色の血痕と、その傍らに置かれた一枚の破れた手紙。かつての栄光と現代の暗殺、その繋ぎは、警視庁の若き探偵、桜井の手に委ねられた。""",
    """في قلب سوق العود المتوهج، حيث تنبعث روائح الفواكه والتوابل من كل ركن، وجدوا جثمانه ملقىً خلف محلّ التوابل القديمة. عيناه مفتوحتان الى الأبد، وكأنّهنّ تحملان أسرار المدينة الضّائعة. قطعة من جلد نادر، غامضة الألوان، وُجدت بالقرب، لتصبح مفتاحاً لغزٍّ يفتكّ سكنة المدينة بأكملها."""
]

futures = {}
results = [None] * len(texts)
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, (lang_id, text) in enumerate(zip(lang_ids, texts)):
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
                    "data": [lang_id],
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/sentencex/infer",
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
                i = futures[future]
                sentences = response.json()["outputs"][0]["data"]
                results[i] = {"lang_id": lang_ids[i], "sentences": sentences}
            except Exception as exc:
                raise ValueError(f"Error getting data from response: {exc} {response.json()}")

print(results)
```

## Performance Analysis
There is some data in [data/sentencex](../data/sentencex/load_sample.json) which can
be used with the `perf_analyzer` CLI in the Triton Inference Server SDK container.

```
sdk-container:/workspace perf_analyzer \
    -m sentencex \
    -v \
    --input-data data/sentencex/load_sample.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 4992.11 infer/sec. Avg latency: 12009 usec (std 6223 usec). 
  * Pass [2] throughput: 4943.35 infer/sec. Avg latency: 12129 usec (std 6504 usec). 
  * Pass [3] throughput: 5042.66 infer/sec. Avg latency: 11896 usec (std 5936 usec). 
  * Client: 
    * Request count: 362198
    * Throughput: 4992.66 infer/sec
    * Avg client overhead: 0.41%
    * Avg latency: 12011 usec (standard deviation 6224 usec)
    * p50 latency: 9689 usec
    * p90 latency: 20033 usec
    * p95 latency: 24419 usec
    * p99 latency: 34945 usec
    * Avg HTTP time: 12006 usec (send 33 usec + response wait 11973 usec + receive 0 usec)
  * Server: 
    * Inference count: 362198
    * Execution count: 7263
    * Successful request count: 362198
    * Avg request latency: 12021 usec (overhead 72 usec + queue 2353 usec + compute input 267 usec + compute infer 8923 usec + compute output 405 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 4992.66 infer/sec, latency 12011 usec

## Validation
The sentencex repo documents their performance against the
[English Golden Rule Set](https://github.com/diasks2/pragmatic_segmenter) so we will
do the same here. Since the validation is listed as either a txt or ruby.spec file, the
data was converted to JSON before doing the evaluation. 

**NOTE** This required a lot of hand editing to convert the files from the ruby spec to JSON.
Also, the sentencex githbub states that they do not include any of the 'list' golden rules.

The data looks like

{
    "en": [
        {
            "title": "Simple period to end sentence #001",
            "text": "Hello World. My name is Jonas.",
            "target": ["Hello World.", "My name is Jonas."],
        },
        {
            "title": "Question mark to end sentence #002",
            "text": "What is your name? My name is Jonas.",
            "target": ["What is your name?", "My name is Jonas."],
        }
    ],
}

### Results
Validation results give 76.2% correct. This differs slightly from the quoted 74.36%.


### Code
The code is available in [model_repository/sentencex/validate.py](../model_repository/sentencex/validate.py)