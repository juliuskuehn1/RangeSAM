FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    libaio-dev \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    python3-setuptools \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# (Optional) Set working directory
WORKDIR /workspace

# Copy your SAM-2 repository into the container
# (Assuming you have it locally alongside the Dockerfile)
COPY sam2 /workspace/
COPY sam2-files/README.md /workspace/sam2-files/
COPY requirements.txt /workspace/
COPY setup.py /workspace/

# Install Python dependencies first (better Docker caching)
RUN pip install --upgrade pip

# Install SAM-2, allowing CUDA build errors
# (SAM2_BUILD_ALLOW_ERRORS is ON by default, but we can be explicit)
ENV SAM2_BUILD_ALLOW_ERRORS=1
RUN pip install -e .

# Fix for MKL threading
ENV MKL_THREADING_LAYER=GNU

# RUN pip install ".[notebooks]" ".[interactive-demo]"
RUN pip uninstall -y typing_extensions
RUN pip uninstall -y typing_extensions
RUN pip install typing_extensions
RUN pip install scikit-learn
RUN pip install open3d
RUN pip install matplotlib
RUN pip install schedulefree
RUN pip install -U catalyst
RUN pip install py-cpuinfo
RUN pip install oneccl-devel
ENV DS_BUILD_FUSED_ADAM=1
RUN pip install deepspeed

COPY . /workspace
# Default command
CMD ["bash"]

