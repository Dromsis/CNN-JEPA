# Image for Ada-Lovelace GPUs (L40S, RTX 40xx, sm_89).
#
# Why this combo:
#   - torch 2.0.1 is the highest torch that pairs cleanly with pytorch-lightning 1.9.x, which
#     this codebase still depends on API-wise (val_dataloaders[0], precision="bf16", ...).
#   - torch 2.0.1 ships prebuilt ONLY as cu11.7 on Docker Hub, and cu11.7 has no sm_89 path.
#     So instead of a pytorch/* base we start from a cu11.8 CUDA base and pip-install the
#     torch 2.0.1 + cu11.8 wheel: its sm_80/sm_86 cubins are forward-compatible to sm_89,
#     so it runs on the L40S.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git curl wget build-essential cmake \
    libsm6 libxrender1 libfontconfig1 libxext6 libgl1 ffmpeg \
    tmux nano htop \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN python -m pip install --upgrade pip

# rclone: syncs the dataset from the Cloudflare R2 bucket (sea-vis-data-fan).
RUN curl https://rclone.org/install.sh | bash || true

# torch 2.0.1 + cu11.8 (runs on L40S sm_89 via sm_80/sm_86 cubin forward-compat).
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Everything on top of torch (lightning 1.9.5, lightly, timm, ...). numpy is pinned <2 there.
COPY requirements-l40s.txt /workspace/requirements-l40s.txt
RUN pip install -r /workspace/requirements-l40s.txt

WORKDIR /workspace
CMD ["/bin/bash"]
