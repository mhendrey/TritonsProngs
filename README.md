This is deprecated and been moved over to [TritonsProngs](https://github.com/TritonsProngs) organization with different repositories for each service.

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

## Translation
Machine translation models translate source text from one language to another. This is
an age-old application of neural networks. Unfortunately, that history seems to cause
some unwanted consequences. Mainly, most machine translation models have been trained
on sentence-level language pairs because that is all the data that the approaches that
the early models could handle. Thus, despite more modern architectures that can support
much large context, the models will stop generating the translation after just a
sentence or two. This means that we also need to provide a way to segment a client's
text into appropriate lengths for the translation model to handle.

The [translate](docs/translate.md) deployment is the main interface that should be
used by most clients. Currently supported models utilized by translate:

* [fastText Language Detection](docs/fasttext_language_identification.md)
  Language identification model. Currently the only available, but future versions may
  include Lingua.
* [Sentencex](docs/sentencex.md)
  Lightweight sentence segmentation. Seems to work well for most languages, with Thai
  and Khmer being noticeable exceptions given their lack of punctutation. Additional
  options like PySBD may be added in the future.
* [SeamlessM4Tv2Large](docs/seamlessm4t_text2text.md)
  Machine translation model that utilizes just the Text-to-Text portion of the
  SeamlessM4T model. Future models will include its predecessor No Language Left
  Behind (NLLB).

## Running Tasks
Running tasks is orchestrated by using [Taskfile.dev](https://taskfile.dev/)

# Taskfile Instructions

This document provides instructions on how to run tasks defined in the `Taskfile.yml`.  

Create a task.env at the root of project to define enviroment overrides. 

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
