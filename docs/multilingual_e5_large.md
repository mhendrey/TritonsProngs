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
To validate that the model is performing as expected, we calculate the performance on the
[BUCC Bitext Mining dataset](https://huggingface.co/datasets/mteb/bucc-bitext-mining)
and compare the performance against results published in the
[Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/abs/2402.05672).

This dataset consists of 4 separate dataset with pairs of sentences in [zh-en, fr-en,
de-en, and ru-en]. Table 5 in the paper reports that the Multilingual E5 large model
achieved **98.6** on this benchmark. Unfortunately the paper doesn't give any details
as to how they did the evaluation. In particular, the BUCC Biitext Mining dataset is
supposed to consist of non-parallel sentences with only about 2-3% of the sentences
having a corresponding translated sentence in the other language. However, the
Huggingface test data has aligned sentences. This may make the task much too easy, but
we will proceed in the absence of more information.

For each language pair dataset, we query with one side and calculate the top-1 accuracy
of finding the corresponding pair in the other language. We calculate a weighted
average across the four sets of language pairs to get a single number. We use
approximate nearest neighbors to perform the search of the 4 nearest neighbors based
upon the cosine distance. We then perform two separate reranking methods before
choosing the top nearest neighbor from this candidate list.  The first is just the
cosine distance itself. The second is based upon a margin scoring approach that is
referenced in the technical approach. This approach is outlined in
[Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings](https://arxiv.org/abs/1811.01136).

The code can be found in the [validate.py](../model_repository/multilingual_e5_large/validate.py)
file.

### Results

| Language Pairs | Margin Accuracy | Cosine Accuracy | # of Records |
| :------------: | :-------------: | :-------------: | :----------: |
| zh-en | 99.53 | 99.26 | 1,899 |
| fr-en | 99.00 | 98.62 | 9,086 |
| de-en | 99.61 | 99.52 | 9,580 |
| ru-en | 97.94 | 97.74 | 14,435|
| **Mean** | **98.76** | **98.54** | |

These match well with the reported 98.6 in the technical report.

### Code
The code is available in [model_repository/multilingual_e5_large/validate.py](../model_repository/multilingual_e5_large/validate.py)