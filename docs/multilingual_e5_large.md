#  Multilingual E5 Text Embeddings
This deployment hosts the [Multilingual-E5-large](https://huggingface.co/intfloat/multilingual-e5-large)
text embedding model. It takes in the input tokens for text and returns the embedding vector
(d=1024) that can be used information retrieval or building downstream classified. The
inputs are integer indices of the tokenizer and the output are float32.

**Note**
  * Maximum input token size is 512 tokens which is the expected size. Set
    'padding="max_length"' when tokenizing to pad appropriately
  * As specified in the Huggingface model card. You need to prefix your text with
    "query: " or "passage: " before tokenizing the text. Note the space after the colon
      * Use "query:" & "passage:" correspondingly for asymmetric tasks such as
        passage retrieval in open QA or ad-hoc information retrieval
      * Use "query:" prefix for symmetric tasks such as semantic similarity, bitext
        mining, paraphrase retrieval
      * Use "query:" prefix if you want to use embeddings as features, such as linear
        probing classification or clustering

Dynamic batching is enabled for this deployment, so clients simply send in each request
separately.

This is a lower level of abstraction, most clients likely should be using
[embed_text](embed_text.md) deployment.

## Example Request
Here's an example request. Just a few things to point out
1. "shape": [1, 512] because we have dynamic batching and the first axis is
   the batch size and the second axis is the maximum token size (pad with '1').
2. "data": this should be "row" flattened. It will be reshaped by the server. Also,
   numpy is not serializable, so convert to python list.

```
import numpy as np
import requests
from transformers import AutoTokenizer

base_url = "http://localhost:8000/v2/models"
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
text = (
    "query: The iridescent chameleon sauntered across the neon-lit cyberpunk cityscape."
)
input_ids = tokenizer(text, padding="max_length")["input_ids"]

inference_request = {
    "inputs": [
        {
            "name": "INPUT_IDS",
            "shape": [1, 512],
            "datatype": "INT64",
            "data": input_ids,
        }
    ]
}
model_response = requests.post(
    url=f"{base_url}/multilingual_e5_large/infer",
    json=inference_request,
).json()

"""
JSON response output looks like
{
    "model_name": "multilingual_e5_large",
    "model_version": "1",
    "outputs": [
        {
            "name": "EMBEDDING",
            "datatype": "FP32",
            "shape": [1, 1024],
            "data": [
                0.01077766,
                -0.0.006316,
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

### Sending Many Requests
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

input_ids_batch = rng.integers(low=0, high=25000, size=[120,512]).astype(np.int64)


futures = {}
embeddings = {}
with ThreadPoolExecutor(max_workers=60) as executor:
    for i, input_ids in enumerate(input_ids_batch):
        inference_request = {
            "inputs": [
                {
                    "name": "INPUT_IDS",
                    "shape": [1, 512],
                    "datatype": "INT64",
                    "data": input_ids.flatten().tolist(),
                }
            ]
        }
        future = executor.submit(requests.post,
            url=f"{base_url}/multilingual_e5_large/infer",
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
There is some data in [data/multilingual_e5_large](../data/multilingual_e5_large/input_ids.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container.

```
sdk-container:/workspace perf_analyzer \
    -m multilingual_e5_large \
    -v \
    --input-data data/multilingual_e5_large/input_ids.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000
```
Gives the following result on an RTX4090 GPU


* Request concurrency: 60
  * Pass [1] throughput: 283.692 infer/sec. Avg latency: 210358 usec (std 43074 usec). 
  * Pass [2] throughput: 285.814 infer/sec. Avg latency: 210039 usec (std 41954 usec). 
  * Pass [3] throughput: 285.9 infer/sec. Avg latency: 209959 usec (std 41938 usec). 
  * Client: 
    * Request count: 20537
    * Throughput: 285.136 infer/sec
    * Avg client overhead: 0.02%
    * Avg latency: 210118 usec (standard deviation 29881 usec)
    * p50 latency: 206747 usec
    * p90 latency: 252549 usec
    * p95 latency: 252905 usec
    * p99 latency: 255893 usec
    * Avg HTTP time: 210110 usec (send 66 usec + response wait 210044 usec + receive 0 usec)
  * Server: 
    * Inference count: 20537
    * Execution count: 857
    * Successful request count: 20537
    * Avg request latency: 209706 usec (overhead 20 usec + queue 125809 usec +
      compute input 150 usec + compute infer 83553 usec + compute output 172 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 285.136 infer/sec, latency 210118 usec



## Validation
**TODO**

**THIS IS JUST THE COPY FORM THE siglip_text**
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
improve peformance. The results below show comparable accuracy.

### Results

|           | Top-1 Accuracy | Top-5 Accuracy | Prompt Template |
|:---------:| :------------: | :------------: | :-------------- |
| Zero-shot | 0.8193         | 0.9630         | A photo of a {text}. |

### Code
The code is available in [model_repository/siglip_text/validate.py](../model_repository/siglip_text/validate.py)