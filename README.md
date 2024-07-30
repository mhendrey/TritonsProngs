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

* [SigLIP](docs/siglip_vision.md) (default)
