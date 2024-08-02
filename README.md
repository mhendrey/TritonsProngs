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

* [SigLIP Text](docs/siglip_text.md) (default)
  Use in conjunction with SigLIP Vision to perform zero-shot learning or semantic
  searching of images with textual descriptions.
