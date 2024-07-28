# Siglip
This deployment hosts the [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
model. It takes in the RGB pixel values for images that have been resized to 384x384
and returns the embedding vector (d=1152) that can be used for zero-/few-shot image
classification. Both the input and output are float32.

Dynamic batching is enabled for this deployment, so clients simply send in a single
image to be embedded.

This is a lower level of abstraction, most clients likely should be using
[embed_image](embed_image.md) deployment.

## Example Request
Here's an example request that just uses random data. Just a few things to point out
1. "shape": [1, 3, 384, 384] because we have dynamic batching and the first axis is
   the batch size, second axis is RGB channels, third/fourth are the heigh/width
2. "data": this should be "row" flattened. It will be reshaped by the server. Also,
   numpy is not serializable, so convert to python list.

```
import numpy as np
import requests

base_url = "http://localhost:8000/v2/models"
rng = np.random.default_rng()

pixel_values_np = rng.normal(
    loc=0.5, scale=0.5, size=[1, 3, 384, 384]
).astype(np.float32)

inference_request = {
    "inputs": [
        {
            "name": "PIXEL_VALUES",
            "shape": [1, 3, 384, 384],
            "datatype": "FP32",
            "data": pixel_values_np.flatten().tolist(),
        }
    ]
}
siglip_response = requests.post(
    url=f"{base_url}/siglip/infer",
    json=inference_request,
).json()

embedding = np.array(
    siglip_response["outputs"][0]["data"],
    dtype=np.float32
)
```

### Sending Many Images
If you want to send a lot of image pixels to be embedded, it's important that you send
each image request in a multithreaded way to achieve optimal throughput. The example
below creates 120 random pixel values to be embedded.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import requests

rng = np.random.default_rng()

pixel_values_batch = rng.normal(
    loc=0.5, scale=0.5, size=[120, 3, 384, 384]
).astype(np.float32)


futures = {}
embeddings = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, pixel_values in enumerate(pixel_values_batch):
        inference_request = {
            "inputs": [
                {
                    "name": "PIXEL_VALUES",
                    "shape": [1, 3, 384, 384],
                    "datatype": "FP32",
                    "data": pixel_values.flatten().tolist(),
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/siglip/infer",
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
                data = response.json()["outputs"][0]["data"]
                embedding = np.array(data, dtype=np.float32)
                embeddings[futures[future]] = embedding
            except Exception as exc:
                raise ValueError(f"Error getting data from response: {exc}")

print(embeddings)
```
## Performance Analysis
There is some data in `data/siglip/pixel_values.json` which can be used with the
`perf_analyzer` CLI in the Triton Inference Server SDK container.

```
sdk-container:/workspace perf_analyzer \
    -m siglip \
    -v \
    --input-data data/siglip/pixel_values.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 110.037 infer/sec. Avg latency: 538142 usec (std 39934 usec). 
  * Pass [2] throughput: 112.479 infer/sec. Avg latency: 535327 usec (std 12405 usec). 
  * Pass [3] throughput: 111.188 infer/sec. Avg latency: 533864 usec (std 7229 usec). 
  * Client: 
    * Request count: 8010
    * Throughput: 111.235 infer/sec
    * Avg client overhead: 0.15%
    * Avg latency: 535768 usec (standard deviation 24455 usec)
    * p50 latency: 532695 usec
    * p90 latency: 552842 usec
    * p95 latency: 562948 usec
    * p99 latency: 582094 usec
    * Avg HTTP time: 535751 usec (send 1142 usec + response wait 534609 usec + receive 0 usec)
  * Server: 
    * Inference count: 8010
    * Execution count: 270
    * Successful request count: 8010
    * Avg request latency: 523216 usec (overhead 48 usec + queue 254808 usec + compute input 8710 usec + compute infer 259454 usec + compute output 194 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 60, throughput: 111.235 infer/sec, latency 535768 usec

## Validation
To validate that the model is performing as expected, we use some data from
[ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge).

Working with images from the training data set, I put 10 images for each of the 1,000
categories into `train/{category_synset}` directory on my local machine. An additional
20 images for each of the 1,000 categories were placed into `valid/{category_synset}".

The training images were embedded using the [embed_image][embed_image.md] Triton
Inference Server deployed endpoint. These embeddings were used to train an
`sklearn.neighbors.KNeighborsClassifier`. Thus, we have a 10-shot learning. To classify
a new embedding vector, the 10 closest neighbors will be used to create a prediction
based upon the class of those 10 neighbors (weighted by distance to query vector).

We calculate the accuracy of the top-1 and find that we get an accuracy of 74.4%

The code is available in `model_repository/siglip/validate.py`.