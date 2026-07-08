# Use NVIDIA CUDA base image for high-performance quantum acceleration
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire framework
COPY . /app

# Install the framework as a Python package
RUN pip3 install -e .

# Set default execution command to run the simulation
CMD ["python3", "run_experiments.py"]
