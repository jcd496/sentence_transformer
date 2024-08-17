# Sentence Transformer Docker Container

This repository provides a Docker container for training and running a sentence transformer model using PyTorch and the `sentence-transformers` library.

## Files

- `Dockerfile`: Contains the instructions to build the Docker image.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `sentence_transformer`: Source code for the project..
    - `train.py`: Python script for training the sentence transformer model.
    - `model.py`: model definition for the sentence transformer model.
    - `data.py`: data loaders and prep for the sentence transformer model.

## Getting Started

1. **Build the Docker Image**

   ```bash
   docker build -t sentence-transformer .

