ARG BASE_IMAGE=continuumio/miniconda3:latest

FROM ${BASE_IMAGE} as builder

#USER conda

ENV CONDA_ENV_NAME=
ENV CONDA_ENV_FILE_PATH=
ENV CONDA_PACKED_OUTPUT_FILE_PATH=

SHELL ["/bin/bash", "-c"]

CMD conda env create -f $CONDA_ENV_FILE_PATH; source /opt/conda/etc/profile.d/conda.sh; conda activate $CONDA_ENV_NAME && conda-pack -o $CONDA_PACKED_OUTPUT_FILE_PATH
