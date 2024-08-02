# SigLIP Text
This deployment hosts the [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
text model. It takes in the input tokens for text and returns the embedding vector
(d=1152) that can be used for zero/few-shot image classification. The input are integer
indices of the tokenizer and the output are float32.

**Note** that the maximum input token size of 64 tokens is much smaller than most
language models. You should think of the text you want embedded as the caption for
an image since that was the kind of data used to train this model.

Dynamic batching is enabled for this deployment, so clients simply send in a single
string of text to be embedded.

This is a lower level of abstraction, most clients likely should be using
[embed_text](embed_text.md) deployment.

## Example Request
Here's an example request. Just a few things to point out
1. "shape": [1, 64] because we have dynamic batching and the first axis is
   the batch size and the second axis is the maximum token size (pad with '1').
2. "data": this should be "row" flattened. It will be reshaped by the server. Also,
   numpy is not serializable, so convert to python list.

```
import numpy as np
import requests
from transformers import SiglipProcessor

base_url = "http://localhost:8000/v2/models"
processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
caption = (
    "A photo of a person in a spacesuit riding a unicorn on the surface of the moon"
)
input_ids_np = processor(caption, padding="max_length")["input_ids"].numpy()

inference_request = {
    "inputs": [
        {
            "name": "INPUT_IDS",
            "shape": [1, 64],
            "datatype": "INT64",
            "data": input_ids_np.flatten().tolist(),
        }
    ]
}
model_response = requests.post(
    url=f"{base_url}/siglip_text/infer",
    json=inference_request,
).json()

"""
JSON response output looks like
{
    "model_name": "siglip_text",
    "model_version": "1",
    "outputs": [
        {
            "name": "EMBEDDING",
            "datatype": "FP32",
            "shape": [1, 1152],
            "data": [
                1.30078125,
                0.61572265,
                ...,
            ]
        }
    ]
}
"""

embedding = np.array(
    model_response["outputs"][0]["data"],
    dtype=np.float32
)
```

### Sending Many Images
If you want to send a lot of text requests to be embedded, it's important that you send
each request in a multithreaded way to achieve optimal throughput. The example
below creates 120 random input ids to be embedded.

NOTE: You will encounter a "OSError Too many open files" if you send a lot of requests.
Typically the default ulimit is 1024 on most system. Either increace this using 
`ulimit -n {n_files}`, or don't create too many futures before you process them when
completed.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import requests

base_url = "http://localhost:8000/v2/models"
rng = np.random.default_rng()

input_ids_batch = rng.integers(low=0, high=32000, size=[120,64]).astype(np.int64)


futures = {}
embeddings = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, input_ids in enumerate(input_ids_batch):
        inference_request = {
            "inputs": [
                {
                    "name": "INPUT_IDS",
                    "shape": [1, 64],
                    "datatype": "INT64",
                    "data": input_ids.flatten().tolist(),
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/siglip_text/infer",
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
There is some data in [data/siglip_text](../data/siglip_vision/input_ids.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container.

```
sdk-container:/workspace perf_analyzer \
    -m siglip_text \
    -v \
    --input-data data/siglip_text/input_ids.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 1569.19 infer/sec. Avg latency: 38205 usec (std 6539 usec). 
  * Pass [2] throughput: 1550.33 infer/sec. Avg latency: 38692 usec (std 6520 usec). 
  * Pass [3] throughput: 1560.74 infer/sec. Avg latency: 38437 usec (std 6436 usec). 
  * Client: 
    * Request count: 112489
    * Throughput: 1560.08 infer/sec
    * Avg client overhead: 0.08%
    * Avg latency: 38444 usec (standard deviation 6501 usec)
    * p50 latency: 40883 usec
    * p90 latency: 42536 usec
    * p95 latency: 42953 usec
    * p99 latency: 44393 usec
    * Avg HTTP time: 38438 usec (send 32 usec + response wait 38406 usec + 
      receive 0 usec)
  * Server: 
    * Inference count: 112489
    * Execution count: 3548
    * Successful request count: 112489
    * Avg request latency: 38271 usec (overhead 32 usec + queue 18085 usec +
      compute input 138 * usec + compute infer 19831 usec + compute output 184 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 1560.08 infer/sec, latency 38444 usec


## Validation
To validate that the model is performing as expected, we use some data from
[ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge).
The training data was nicely organized into subdirectories with each subdirectory
named after the Synset category and with each file name in a give subdirectory also
containing the {synset}_{file_id}.JPEG.

Working with images from the training data set, I put 10 images for each of the 1,000
categories into `train/{synset}` directory on my local machine. An additional
20 images for each of the 1,000 categories were placed into `valid/{synset}`.

```
train/
  - n01440764/
    - n01440764_3198.JPEG
    - n01440764_3199.JPEG
    - ...
  - n01443537/
    - n01443537_428.JPEG
    - ...
  - ...
```

In addition to the subset of images, I also downloaded the LOC_synset_mapping.txt. This
contains the synset category label and a description of the category. This data will be
used for performing the zero-shot accuracy validation. Here is the first
few lines:

| Label | Text Description |
| :----: | :-----------|
| n01440764 | tench, Tinca tinca |
| n01443537 | goldfish, Carassius auratus |
| n01484850 | great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias |
| n01491361 | tiger shark, Galeocerdo cuvieri |
| n01494475 | hammerhead, hammerhead shark |

### Zero-Shot KNN Classifier
The [SigLIP paper](https://arxiv.org/abs/2303.15343) uses zero-shot to measure the
quality of their embedding model. For zero-shot, you use a text description of the
category and embed that using the
[SiglipTextModel](https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipTextModel).
This text embedding of the category is what is used to fit a KNN Classifier. Taking
each validation image embedding, get the 10 nearest neighbors (where a neighbor is a
text embedding of a category), and use the neighbors' corresponding category label to
predict the classification of the validation image. We calculate both the top-1 and
top-5 accuracy where top-k means the classifier was correct if the true label appears
among the top k predicted category labels.

The [siglip_vision][siglip_vision.md] deployment is used to embed the validation
images.

The SigLIP paper claims an ImageNet accuracy of 83.2% on the validation data of
ImageNet. The paper notes some tweak to the prompts and a few other details to
improve peformance. The numbers quoted below had just a few rounds of iterating
on the prompt to use.

### Results

|           | Top-1 Accuracy | Top-5 Accuracy | Prompt Template |
|:---------:| :------------: | :------------: | :-------------- |
| Zero-shot | 0.7556         | 0.9211         | This is a photo from ImageNet's {label} category. This category contains photos of {text}. |

### Code
The code is available in [model_repository/siglip_text/validate.py](../model_repository/siglip_text/validate.py)