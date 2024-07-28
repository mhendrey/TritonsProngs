# Embed Image
This is a BLS deployment that lets a client send an image and get back a vector
embedding. Currently this only uses the [siglip](siglip.md) model, but future
embedding models could be added.

Because dynamic batching has been enabled for these Triton Inference Server
deployments, clients simply send each request separately. This simplifies the code for
the client, see examples below, yet they reap the benefits of batched processing. In
addition, this allows for controlling the GPU RAM consumed by the server.

## Example Request
### Raw Image
Here's an example of sending a raw image. Just a few things to point out

1. Must have the 'headers' argument specified. The 'Content-Type' is normal, but
   the 'Inference-Header-Content-Length' is specific to Triton. Set this to zero
   and it will calculate the length of the header for you and return that back.
2. The response that comes back are bytes with the header information first
   concatenated with the bytes of the embedding vector

```
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"

image_path = "/path/to/your/image.png"
with open(image_path, "rb") as f:
    image_bytes = f.read()

image_embed_response = requests.post(
    url=f"{base_url}/embed_image/infer",
    headers={
        "Content-Type": "application/octet-stream",
        "Inference-Header-Content-Length": "0",
    },
    data=image_bytes,
)

header_length = int(image_embed_response.headers["Inference-Header-Content-Length"])
image_embedding_np = np.frombuffer(
    image_embed_response.content[header_length:],
    dtype=np.float32,
)
```

### Base64 Encoded Image
Here's an example of sending a base64 encoded image. Just a few things to point out

1. decode("UTF-8") the b64encoded bytes to give you a `str` so JSON doesn't get mad
2. Must use "parameters" and set "base64_encoded" to `True`. The default is `False`
3. "shape": [1, 1] because we have dynamic batching enabled and the first axis is batch
   size

```
import base64
import requests

base_url = "http://localhost:8000/v2/models"

image_path = "/path/to/your/image.png"
with open(image_path, "rb") as f:
    image_bytes = f.read()
image_b64_str = base64.b64encode(image_bytes).decode("UTF-8")

inference_request = {
    "parameters": {"base64_encoded": True},  # **MUST SET THIS**
    "inputs": [
        {
            "name": "INPUT_IMAGE",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [image_b64_str],
        }
    ]
}
image_embed_response = requests.post(
    url=f"{base_url}/embed_image/infer",
    json=inference_request,
).json()
image_embedding = image_embed_response["outputs"][0]["data"]
```

### Sending Many Images
If you want to send a lot of images to be embedded, it's important that you send each
image request in a multithreaded way to achieve optimal throughput.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import requests

input_dir = Path("/path/to/image/director/")
futures = {}
embeddings = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for path in input_dir.iterdir():
        if path.is_file():
            future = executor.submit(requests.post,
                url=f"{base_url}/embed_image/infer",
                data=path.read_bytes(),
                headers={
                    "Content-Type": "application/octet-stream",
                    "Inference-Header-Content-Length": "0",
                }
            )
            futures[future] = str(path.absolute())
    
    for future in as_completed(futures):
        try:
            response = future.result()
        except Exception as exc:
            print(f"{futures[future]} threw {exc}")
        else:
            try:
                header_length = int(response.headers["Inference-Header-Content-Length"])
                embedding = np.frombuffer(
                    response.content[header_length:], dtype=np.float32
                )
                embeddings[futures[future]] = embedding
            except Exception as exc:
                raise ValueError(f"Error getting data from response: {exc}")
print(embeddings)
```
## Performance Analysis
There is some data in `data/embed_image/base64.json` which can be used with the
`perf_analyzer` CLI in the Triton Inference Server SDK container. The only issue is
that the `base64_encoded` request parameter needs to be set to `true` (default is
`false`). This can be done as an option for `perf_analyzer`, but only if used in
conjunction with `-i grpc`.

```
sdk-container:/workspace perf_analyzer \
    -m embed_image \
    -i grpc \
    -v \
    --input-data data/embed_image/base64.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000 \
    --bls-composing=siglip \
    --request-parameter=base64_encoded:true:bool
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 108.827 infer/sec. Avg latency: 547194 usec (std 44242 usec). 
  * Pass [2] throughput: 108.265 infer/sec. Avg latency: 545026 usec (std 66473 usec). 
  * Pass [3] throughput: 110.269 infer/sec. Avg latency: 549658 usec (std 79885 usec). 
  * Client: 
    * Request count: 7858
    * Throughput: 109.121 infer/sec
    * Avg client overhead: 0.01%
    * Avg latency: 547307 usec (standard deviation 43777 usec)
    * p50 latency: 543346 usec
    * p90 latency: 587343 usec
    * p95 latency: 920179 usec
    * p99 latency: 1007879 usec
    * Avg gRPC time: 547295 usec (marshal 16 usec + response wait 547279 usec + unmarshal 0 usec)
  * Server: 
    * Inference count: 7858
    * Execution count: 144
    * Successful request count: 7858
    * Avg request latency: 547622 usec (overhead 379345 usec + queue 60917 usec + compute 107360 usec)
  * Composing models: 
  * siglip, version: 1
      * Inference count: 7874
      * Execution count: 1019
      * Successful request count: 7874
      * Avg request latency: 168289 usec (overhead 12 usec + queue 60917 usec + compute input * 1967 usec + compute infer 105296 usec + compute output 96 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 109.121 infer/sec, latency 547307 usec