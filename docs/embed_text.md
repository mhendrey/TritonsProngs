# Embed Text
This is a BLS deployment that lets a client send text and get back a vector
embedding. Currently this supports both the [Multilingual E5](multilingual_e5_large.md)
and the [SigLIP Text](siglip_text.md) model. Multilingual E5 is the default
`embed_model`. In order to use the SigLIP text embedding model, you must specify that
as a request parameter (see examples below)

Because dynamic batching has been enabled for these Triton Inference Server
deployments, clients simply send each request separately. This simplifies the code for
the client, see examples below, yet they reap the benefits of batched processing. In
addition, this allows for controlling the GPU RAM consumed by the server.

## Multilingual E5 Text Embeddings
For optimal performance all text sent should have either "query: " or "passage: "
prepended to your text. Notice that there is a space after the colon. See the **NOTE**
in [Multilingual E5](multilingual_e5_large.md) about when to use one vs the other.

### Send Single Request
```
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"

text = (
    "query: The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape."
)
inference_json = {
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [text],
        }
    ]
}
text_embed_response = requests.post(
    url=f"{base_url}/embed_text/infer",
    json=inference_json,
)

response_json = text_embed_response.json()
"""
{
    "model_name": "embed_text",
    "model_version": "1",
    "outputs": [
        {
            "name": "EMBEDDING",
            "shape": [1, 1024],
            "datatype": "FP32",
            "data": [
                0.00972023,
                -0.00810159,
                -0.00420804,
                ...,
            ]
        }
    ]
}
"""
text_embedding = np.array(response_json["outputs"][0]["data"]).astype(np.float32)
```

### Sending Many Requests
When embedding multiple text strings, use multithreading to send requests in parallel,
which maximizes throughput and efficiency.

NOTE: You will encounter an "OSError: Too many open files" if you send a lot of
requests. Typically the default ulimit is 1024 on most system. Either increase this
using `ulimit -n {n_files}`, or don't create too many futures before you processing
some of them.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import requests

base_url = "http://localhost:8000/v2/models"

texts = [
    "query: Did Palpatine die in The Return of the Jedi?",
    "query: What causes ocean tides?",
    "query: How does photosynthesis work in plants?",
    "query: Explain the concept of supply and demand in economics.",
    "query: What is the difference between weather and climate?",
    "query: How does the human immune system defend against pathogens?",
    "query: How are artificial intelligence models created?",
]

futures = {}
embeddings = {}
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
            url=f"{base_url}/embed_text/infer",
            json=infer_request,
        )
        futures[future] = i
    
    for future in as_completed(futures):
        try:
            response = future.result()
        except Exception as exc:
            print(f"{futures[future]} threw {exc}")
        try:
            embedding = response.json()["outputs"][0]["data"]
            embedding = np.array(embedding).astype(np.float32)
        except Exception as exc:
            raise ValueError(f"Error getting data from response: {exc}")
        embeddings[futures[future]] = embedding
print(embeddings)
```

### Performance Analysis
There is some data in [data/embed_text](../data/embed_text/multilingual_text.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container to measure the throughput of the Multilingual E5 Text model.

```
sdk-container:/workspace perf_analyzer \
    -m embed_text \
    -v \
    --input-data data/embed_text/multilingual_text.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000 \
    --bls-composing=multilingual_e5_large
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 267.873 infer/sec. Avg latency: 222976 usec (std 44315 usec). 
  * Pass [2] throughput: 272.654 infer/sec. Avg latency: 220940 usec (std 44568 usec). 
  * Pass [3] throughput: 265.379 infer/sec. Avg latency: 224063 usec (std 44111 usec). 
  * Client: 
    * Request count: 19347
    * Throughput: 268.635 infer/sec
    * Avg client overhead: 0.01%
    * Avg latency: 222645 usec (standard deviation 7743 usec)
    * p50 latency: 217243 usec
    * p90 latency: 229589 usec
    * p95 latency: 258700 usec
    * p99 latency: 425550 usec
    * Avg HTTP time: 222639 usec (send 47 usec + response wait 222592 usec + receive 0 usec)
  * Server: 
    * Inference count: 19347
    * Execution count: 341
    * Successful request count: 19347
    * Avg request latency: 222730 usec (overhead 135916 usec + queue 30328 usec + compute 56486 usec)

  * Composing models: 
  * multilingual_e5_large, version: 1
      * Inference count: 19395
      * Execution count: 1890
      * Successful request count: 19395
      * Avg request latency: 86828 usec (overhead 14 usec + queue 30328 usec + compute input 74 usec + compute infer 56306 usec + compute output 105 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 268.635 infer/sec, latency 222645 usec


## SigLIP Text Embedding Model
Since this is not the default `embed_model`, you must pass it explicitly as a request
parameter. Otherwise everything works the same way. This model is useful in conjunction
with the [SigLIP Vision](siglip_vision.md) to enable image search via natural language
or performing zero-shot image classification.

### Sending Single Request
Here's an example of sending a string of text.

```
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"

text = (
    "A photo of a person in an astronaut suit riding a "
    + "unicorn on the surface of the moon."
)
inference_json = {
    "parameters": {"embed_model": "siglip_text"},
    "inputs": [
        {
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [text],
        }
    ]
}
text_embed_response = requests.post(
    url=f"{base_url}/embed_text/infer",
    json=inference_json,
)

response_json = text_embed_response.json()
"""
{
    "model_name": "embed_text",
    "model_version": "1",
    "outputs": [
        {
            "name": "EMBEDDING",
            "shape": [1, 1152],
            "datatype": "FP32",
            "data": [
                1.19238,
                0.90869,
                0.44360,
                ...,
            ]
        }
    ]
}
"""
text_embedding = np.array(response_json["outputs"][0]["data"]).astype(np.float32)

```

### Sending Many Requests
When embedding multiple text strings, use multithreading to send requests in parallel,
which maximizes throughput and efficiency.

NOTE: You will encounter an "OSError: Too many open files" if you send a lot of
requests. Typically the default ulimit is 1024 on most system. Either increase this
using `ulimit -n {n_files}`, or don't create too many futures before you processing
some of them.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import requests

base_url = "http://localhost:8000/v2/models"

alt_texts = [
    "Close-up of a student sitting next to her practice piano.",
    "Toddler excitedly eating a slice of cheese pizza larger than her head!",
    "Wide angle from above showing the entire garden, guests, and wedding party as the bride walks down the aisle.",
    "A smiling, elderly mother places her hand on her daughter’s cheek as she prepares for the wedding ceremony.",
    "An embracing couple in formal attire smile as they look at a Rainbow lorikeet perched on the man’s finger.",
    "High school senior in blue jeans and a t-shirt sits on a rock for his senior portraits.",
]

futures = {}
embeddings = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, alt_text in enumerate(alt_texts):
        infer_request = {
            "parameters": {"embed_model": "siglip_text"},
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [alt_text],
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/embed_text/infer",
            json=infer_request,
        )
        futures[future] = i
    
    for future in as_completed(futures):
        try:
            response = future.result()
        except Exception as exc:
            print(f"{futures[future]} threw {exc}")
        try:
            embedding = response.json()["outputs"][0]["data"]
            embedding = np.array(embedding).astype(np.float32)
        except Exception as exc:
            raise ValueError(f"Error getting data from response: {exc}")
        embeddings[futures[future]] = embedding
print(embeddings)
```
### Performance Analysis
There is some data in [data/embed_text](../data/embed_text/imagenet_categories.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container. Because we need to send the request parameter, we must specify using the
gRPC protocol which supports this option in `perf_analyzer`.

```
sdk-container:/workspace perf_analyzer \
    -m embed_text \
    -v \
    -i grpc \
    --input-data data/embed_text/imagenet_categories.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000 \
    --bls-composing=siglip_text \
    --request-parameter=embed_model:siglip_text:string
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 1183.76 infer/sec. Avg latency: 50612 usec (std 4734 usec). 
  * Pass [2] throughput: 1180.5 infer/sec. Avg latency: 50878 usec (std 4474 usec). 
  * Pass [3] throughput: 1183.16 infer/sec. Avg latency: 50653 usec (std 4160 usec). 
  * Client: 
    * Request count: 85231
    * Throughput: 1182.47 infer/sec
    * Avg client overhead: 0.07%
    * Avg latency: 50715 usec (standard deviation 4463 usec)
    * p50 latency: 49712 usec
    * p90 latency: 55982 usec
    * p95 latency: 59582 usec
    * p99 latency: 67712 usec
    * Avg gRPC time: 50705 usec (marshal 2 usec + response wait 50703 usec + unmarshal 0 usec)
  * Server: 
    * Inference count: 85231
    * Execution count: 1422
    * Successful request count: 85231
    * Avg request latency: 51241 usec (overhead 35390 usec + queue 6511 usec + compute 9340 usec)

  * Composing models: 
  * siglip_text, version: 1
      * Inference count: 85260
      * Execution count: 8536
      * Successful request count: 85260
      * Avg request latency: 15861 usec (overhead 10 usec + queue 6511 usec + compute input 60 usec + compute infer 9202 usec + compute output 77 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 1182.47 infer/sec, latency 50715 usec