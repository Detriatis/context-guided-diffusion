FROM continuumio/miniconda3

# Copy environment file if you have one
COPY ./materials/environment.yml /tmp/env.yml
COPY ./small_molecules/environment.yml /tmp/env2.yml

# Create the environment
RUN conda env create -f /tmp/env.yml
RUN conda env create -f /tmp/env2.yml

# Activate the environment by default
SHELL ["/bin/bash", "-c"]

# Set working directory
WORKDIR /workspace

# Install additional system tools if needed
RUN apt-get update && apt-get install -y \
    build-essential git curl

# Optional: expose ports for Jupyter
EXPOSE 8888

