# SigLIP Vision
This deployment hosts the [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
vision model. It takes in the RGB pixel values for images that have been resized to 384x384
and returns the embedding vector (d=1152) that can be used for zero/few-shot image
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
siglip_vision_response = requests.post(
    url=f"{base_url}/siglip_vision/infer",
    json=inference_request,
).json()

embedding = np.array(
    siglip_vision_response["outputs"][0]["data"],
    dtype=np.float32
)
```

### Sending Many Images
If you want to send a lot of image pixels to be embedded, it's important that you send
each image request in a multithreaded way to achieve optimal throughput. The example
below creates 120 random pixel values to be embedded.

NOTE: You will encounter a OSError Too many open files if you send a lot of requests.
Typically the default ulimit is 1024 on most system. Either increaces this using 
`ulimit -n {n_files}`, or don't create too many futures before you process them when
completed.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import requests

base_url = "http://localhost:8000/v2/models"
rng = np.random.default_rng()

pixel_values_batch = rng.normal(
    loc=0.5, scale=0.5, size=[60, 3, 384, 384]
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
            url=f"{base_url}/siglip_vision/infer",
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
There is some data in [data/siglip_vision](../data/siglip_vision/pixel_values.json)
which can be used with the `perf_analyzer` CLI in the Triton Inference Server SDK
container.

```
sdk-container:/workspace perf_analyzer \
    -m siglip_vision \
    -v \
    --input-data data/siglip_vision/pixel_values.json \
    --measurement-mode=time_windows \
    --measurement-interval=20000 \
    --concurrency-range=60 \
    --latency-threshold=1000
```
Gives the following result on an RTX4090 GPU

* Request concurrency: 60
  * Pass [1] throughput: 121.352 infer/sec. Avg latency: 488315 usec (std 21956 usec). 
  * Pass [2] throughput: 122.477 infer/sec. Avg latency: 487771 usec (std 4739 usec). 
  * Pass [3] throughput: 123.644 infer/sec. Avg latency: 486538 usec (std 4041 usec). 
  * Client: 
    * Request count: 8821
    * Throughput: 122.491 infer/sec
    * Avg client overhead: 0.16%
    * Avg latency: 487536 usec (standard deviation 13141 usec)
    * p50 latency: 486777 usec
    * p90 latency: 494846 usec
    * p95 latency: 497160 usec
    * p99 latency: 523129 usec
    * Avg HTTP time: 487520 usec (send 979 usec + response wait 486541 usec + receive 0 usec)
  * Server: 
    * Inference count: 8821
    * Execution count: 295
    * Successful request count: 8821
    * Avg request latency: 478045 usec (overhead 25 usec + queue 233130 usec + compute input 8070 usec + compute infer 236598 usec + compute output 221 usec)

* Inferences/Second vs. Client Average Batch Latency
* Concurrency: 60, throughput: 122.491 infer/sec, latency 487536 usec

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

### 10-Shot Training of KNN Classifier
As a first check, we will use the training images (10 images per category x 1000
categories) to create a [KNN Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier).
The training images are embedded using the [embed_image][embed_image.md] Triton
Inference Server deployed endpoint. These training image embeddings and their
corresponding category labels were used to fit the classifier.

The validation images are used to measure the performance. Each validation image is
also sent to be embedded. For each image, the classifier finds the 10-nearest training
images. A prediction for the classification of the validation image is based upon the
category of the 10 nearest neighbors and their distance to the validation image. Both
the top-1 accuracy (i.e., was the true label of the validation image the top predicted
class from the nearest neighbors) and also the top-5 accuracy (was the true label in
among the top-5 predicted classes). Results are in the table below.

### Zero-Shot KNN Classifier
The [SigLIP paper](https://arxiv.org/abs/2303.15343) uses zero-shot to measure the
quality of their embedding model. For zero-shot, you use a text description of the
category and embed that using the
[SiglipTextModel](https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipTextModel).
This text embedding of the category is what is used to fit the KNN Classifier. After
that, we do the same as before. Taking each validation image embedding, get the 
10 nearest neighbors (where a neighbor now is a text embedding of a category), and
use the neighbors' corresponding category label to predict the classification of the
validation image. We calculate both the top-1 and top-5 accuracy.

The SigLIP paper claims an ImageNet accuracy of 83.2% on the validation data of
ImageNet. They paper notes some tweak to the prompts and a few other details to
improve peformance. The numbers quoted below show a few different variations of
prompting templates and demonstrate comparable accuracy to the paper.

Interesting to note significant improvement in these scores when `padding="max_length"`
is set when calling the processor to tokenize the text. I have no explanation why,
but the Huggingface model card does explicitly call this out. Without padding, top-1
accuracy falls from 0.8194 -> 0.5142 and top-5 accuracy falls from 0.9630 ->
0.7525.

### Results

|           | Top-1 Accuracy | Top-5 Accuracy | Prompt Template |
|:---------:| :------------: | :------------: | :-------------- |
|   10-shot | 0.7448         | 0.9153         |                 |
| Zero-shot | 0.8194         | 0.9630         | A photo of {text}. |
| Zero-shot | 0.8063         | 0.9550         | This is a photo containing images of {text}. |
| Zero-shot | 0.7558         | 0.9210         | This is a photo from ImageNet's {label} category. This category contains photos of {text}. |

### Code
The code is available in [model_repository/siglip_vision/validate.py](../model_repository/siglip_vision/validate.py)