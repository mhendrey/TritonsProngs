# TritonsProngs
Collection of Triton Inference Server deployment packages.

## Image Embedding
Image embedding is the process of taking an image (pixel values) and running them through
an embedding model which returns a vector representation of the image. This allows for
the following example application:

1. Find images that are similar to a query image
2. Use the vectors as a starting point for building an image classifier. This allows
   for making much smaller models that need a lot less labeled example because the
   the model creator can start from a much richer representation of the image.
3. Image clustering to combine images together that are similar to allow for faster
   human reviewing

The [embed_image](docs/embed_image.md) Triton Inference Server deployment allows
clients to send either the raw bytes of an image or a JSON request of the base64
encoded image. Current supported models

* [SigLIP](docs/siglip.md) (default)
