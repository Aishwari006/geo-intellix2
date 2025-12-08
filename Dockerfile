# ==================================================
# ISRO GEO-INTELLIX: PRODUCTION DOCKERFILE (OPTIMIZED)
# ==================================================

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# STEP 1: System Dependencies
# Removed 'libgl1' because we are using opencv-headless
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3-pip \
    git build-essential \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# STEP 2: Setup Workspace
WORKDIR /app/GeoChat

# STEP 3: Install Dependencies (The Caching Layer)
# We ONLY copy requirements first. This way, changing code doesn't trigger a reinstall.
COPY requirements.txt .

# Combined Pinning and Installing to reduce layers
# 1. Pin NumPy < 2.0 (Critical)
# 2. Install PyTorch (Heavy)
# 3. Install the rest from requirements.txt
RUN pip install --no-cache-dir "numpy<2.0" && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 deepspeed==0.9.5 && \
    pip install --no-cache-dir -r requirements.txt --upgrade

# STEP 4: Copy Application Code
# We do this LAST so builds are fast
COPY . .

# Install the local package (GeoChat)
RUN pip install .

# STEP 5: Runtime
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

# STEP 6: Install Streamlit
RUN pip install streamlit
CMD ["python", "geochat_demo.py", "--model-path", "geochat-7B", "--load-4bit"]