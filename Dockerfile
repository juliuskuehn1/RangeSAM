# Start from an official PyTorch image with CUDA (for GPU builds)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Set working directory
WORKDIR /workspace

# Copy your SAM-2 repository into the container
# (Assuming you have it locally alongside the Dockerfile)
COPY . /workspace

# Install Python dependencies first (better Docker caching)
RUN pip install --upgrade pip

# Install SAM-2, allowing CUDA build errors
# (SAM2_BUILD_ALLOW_ERRORS is ON by default, but we can be explicit)
ENV SAM2_BUILD_ALLOW_ERRORS=1
RUN pip install -e .

# RUN pip install ".[notebooks]" ".[interactive-demo]"
RUN pip uninstall -y typing_extensions
RUN pip uninstall -y typing_extensions
RUN pip install typing_extensions
RUN pip install scikit-learn
RUN pip install open3d
RUN pip install matplotlib
COPY train.py .
COPY SAM2UNet.py .
# Default command
CMD ["bash"]

