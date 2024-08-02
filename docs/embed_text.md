# Embed Text
This is a BLS deployment that lets a client send text and get back a vector
embedding. Currently this only uses the [SigLIP Text](siglip_text.md) model, but
future embedding models will be added.

Because dynamic batching has been enabled for these Triton Inference Server
deployments, clients simply send each request separately. This simplifies the code for
the client, see examples below, yet they reap the benefits of batched processing. In
addition, this allows for controlling the GPU RAM consumed by the server.

## Example Request
### Text
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

### Sending Many Strings of Text
If you want to send a lot of strings of text to be embedded, it's important that
you send each request in a multithreaded way to achieve optimal throughput.

NOTE: You will encounter an "OSError: Too many open files" if you send a lot of
requests. Typically the default ulimit is 1024 on most system. Either increaces this
using `ulimit -n {n_files}`, or don't create too many futures before you process them
when completed.

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
## Performance Analysis
There is some data in [data/embed_text](../data/embed_text/imagenet_categories.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container.

```
sdk-container:/workspace perf_analyzer \
    -m embed_text \
    -v \
    --input-data data/embed_text/imagenet_categories.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000 \
    --bls-composing=siglip_text
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 1236.92 infer/sec. Avg latency: 48428 usec (std 9910 usec). 
  * Pass [2] throughput: 1274.63 infer/sec. Avg latency: 47118 usec (std 10006 usec). 
  * Pass [3] throughput: 1261.65 infer/sec. Avg latency: 47519 usec (std 9964 usec). 
  * Client: 
    * Request count: 90652
    * Throughput: 1257.74 infer/sec
    * Avg client overhead: 0.07%
    * Avg latency: 47681 usec (standard deviation 9975 usec)
    * p50 latency: 45445 usec
    * p90 latency: 56111 usec
    * p95 latency: 65939 usec
    * p99 latency: 90062 usec
    * Avg HTTP time: 47674 usec (send 39 usec + response wait 47635 usec + receive 0 usec)
  * Server: 
    * Inference count: 90652
    * Execution count: 1577
    * Successful request count: 90652
    * Avg request latency: 47816 usec (overhead 33025 usec + queue 5964 usec + compute 8827 usec)

  * Composing models: 
  * siglip_text, version: 1
      * Inference count: 90681
      * Execution count: 8885
      * Successful request count: 90681
      * Avg request latency: 14801 usec (overhead 10 usec + queue 5964 usec + compute input 59 usec + compute infer 8695 usec + compute output 72 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 1257.74 infer/sec, latency 47681 usec
