# TritonsProngs
Collection of Triton Inference Server deployment packages.

## Image Embedding
The [embed_image](docs/embed_image.md) allows clients to send either the raw bytes
of an image or a JSON request of the base64 encoded image. Under the hood this uses
the [SigLIP](docs/siglip.md) model to return the 1-d vector representation of image.