ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim

# Avoid interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive

ARG ESP_PPQ_REPO=https://github.com/espressif/esp-ppq.git
ARG ESP_PPQ_REF=4168722

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
        bash \
        cmake \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU. Try the CPU index first (x86_64),
# fall back to default PyPI (aarch64 has wheels on PyPI).
RUN python -m pip install --upgrade --no-cache-dir pip setuptools wheel \
    && (python -m pip install --no-cache-dir \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            'torch>=2.0,<3' 'torchvision>=0.15' 'torchaudio>=2.0' \
        || python -m pip install --no-cache-dir \
            'torch>=2.0,<3' 'torchvision>=0.15' 'torchaudio>=2.0')

# Core deps for esp-ppq (match its pyproject constraints)
RUN python -m pip install --no-cache-dir \
        'numpy>=1.24.0' \
        'onnx>=1.14.0,<1.18.0' \
        'onnxruntime>=1.19.0' \
        'onnxsim>=0.4.36' \
        'tqdm>=4.67.1' \
        'cryptography>=45.0.4' \
        'flatbuffers>=25.2.10'

# Clone esp-ppq at a pinned commit and install from source.
WORKDIR /opt
RUN git clone "${ESP_PPQ_REPO}" /opt/esp-ppq \
    && cd /opt/esp-ppq \
    && git checkout "${ESP_PPQ_REF}" \
    && rm -rf .git
RUN python -m pip install --no-cache-dir /opt/esp-ppq

# Default workdir where the repo will be mounted
WORKDIR /work

# Default cmd is Python; the exporter will provide the script/args
ENTRYPOINT ["python"]
CMD ["--version"]
