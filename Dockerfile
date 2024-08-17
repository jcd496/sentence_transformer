# Dockerfile

# Use the official PyTorch image as the base image
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script into the container
COPY sentence_transformer .

# Run the Python script
CMD ["python", "train.py"]

