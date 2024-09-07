# TritonsProngs
Collection of Triton Inference Server deployment packages.

## Image Embedding
Image embedding is a technique that transforms visual information from an image into a
compact numerical representation, typically in the form of a fixed-length vector. A
good representation captures essential features and characteristics of the image,
allowing for efficient processing and comparison of visual data in various machine
learning and computer vision tasks. Some common use cases for image embeddings include:

* Transfer learning for image classification
  * Allows for making smaller downstream models that need less labeled data
* Image search
  * Find images similar to a known starting image
  * Find images by giving textual descriptions
* Face recognition and verification
* Image clustering and categorization

The [embed_image](docs/embed_image.md) Triton Inference Server deployment allows
clients to send either the raw bytes of an image or a JSON request of the base64
encoded image. Current supported models:

* [SigLIP Vision](docs/siglip_vision.md) (default)

## Text Embedding
Text embedding models convert text into dense numerical vectors, capturing semantic
meaning in a high-dimensional space. These vector representations enable machines to
process and understand textual data more effectively, facilitating various natural
language processing tasks.

* Document clustering and classification
    * Allows for making smaller downstream models that need less labeled data
* Semantic search and information retrieval
    * When paired with corresponding image embedding, enables searching for images by writing the alt-text description.
* Question/Answering Systems

The [embed_text](docs/embed_text.md) deployment is the main interface that should be
used by most clients. Currently supported models accessible within embed_text:

* [Multilingual E5 Text](docs/multilingual_e5_large.md) (default)
  Trained specifically to support multilingual retrieval capabilities, cross-lingual
  similarity search, and multilingual document classification.
* [SigLIP Text](docs/siglip_text.md)
  Use in conjunction with SigLIP Vision to perform zero-shot learning or semantic
  searching of images with textual descriptions.

## Running Tasks
Running tasks is orchestrated by using [Taskfile.dev](https://taskfile.dev/)

# Taskfile Instructions

This document provides instructions on how to run tasks defined in the `Taskfile.yml`.  

Create a `.env` at the root of project to define enviroment overrides. 

## Tasks Overview

The `Taskfile.yml` includes the following tasks:

- `triton-start`
- `triton-stop`
- `model-import`
- `build-execution-env-all`
- `build-*-env` (with options: `embed_image`, `embed_text`, `siglip_vision`, `siglip_text`, `multilingual_e5_large`)

## Task Descriptions

### `triton-start`

Starts the Triton server.

```sh
task triton-start
```

### `triton-stop`

Stops the Triton server.

```sh
task triton-stop
```

### `model-import`

Import model files from huggingface

```sh
task model-import
```

### `task build-execution-env-all`

Builds all the conda pack environments used by Triton

```sh
task build-execution-env-all
```

### `task build-*-env`

Builds specific conda pack environments used by Triton

```sh
#Example 
task build-siglip_text-env
```

### `Complete Order`

Example of running multiple tasks to stage items needed to run Triton Server

```sh
task build-execution-env-all
task model-import
task triton-start
# Tail logs of running containr
docker logs -f $(docker ps -q --filter "name=triton-inference-server")
```

### `Overriding Defaults`  

At Root of project define file *local.env* 

### Environment Variables for `local.env`

The following docker compose and Task variables can be overridden in the `local.env` file:

| Variable Name     | Description                                                      | Default Value                   |
|-------------------|------------------------------------------------------------------|---------------------------------|
| `TRITON_INFERENCE_SERVER_IMAGE`    | TRITON BASE IMAGE                               | `nvcr.io/nvidia/tritonserver`   |
| `TRITON_INFERENCE_SERVER_TAG`      | TRITON BASE IMAGE VERSION                       | `24.07-py3`                     |
| `MODEL_REPOSITORY_SOURCE`          | SOURCE MODEL REPO DIR                           | `./model-repository`            |
| `MODEL_REPOSITORY_TARGET`          | TARGET MODEL REPO DIR                           | `/model-repository`             |
| `HF_HUB_CACHE_SOURCE`              | SOURCE HUGGING FACE CACHE DIR                   | `./models`                      |
| `HF_HUB_CACHE_TARGET`              | TARGET HUGGING FACE CACHE DIR                   | `/models`                       |
| `COMPOSE_CONDA_PACK_BASE_IMAGE`    | COMPOSE CONDA PACK BASE IMAGE                   | `continuumio/miniconda3:latest` |

**Note:** Format in `local.env` file is as following `VAR`=`VALUE`.
