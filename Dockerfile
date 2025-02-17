FROM nvidia/cuda:12.1.0-base-ubuntu22.04

WORKDIR /comfyui

# Combine RUN commands to reduce layers and use DEBIAN_FRONTEND to prevent interactive prompts
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    libglib2.0-0 \
    libgl1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clone repository and update submodules
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui && \
    cd /comfyui && \
    git submodule update --init --recursive

# Set up Python virtual environment and install dependencies
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install opencv-python && \
    pip install onnxruntime-gpu insightface && \
    pip install -r requirements.txt

RUN cd /comfyui/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
RUN cd /comfyui/custom_nodes && git clone https://codeberg.org/Gourieff/comfyui-reactor-node.git
RUN cd /comfyui/custom_nodes/comfyui-reactor-node && . /comfyui/venv/bin/activate && pip install -r requirements.txt && python3 install.py
RUN cd /comfyui/models && mkdir -p ultralytics/bbox && cd ultralytics/bbox && wget https://huggingface.co/datasets/Gourieff/ReActor/raw/main/models/detection/bbox/face_yolov8m.pt

RUN cd /comfyui/custom_nodes && git clone https://github.com/mav-rik/facerestore_cf.git
RUN cd /comfyui/custom_nodes/facerestore_cf && . /comfyui/venv/bin/activate && pip3 install -r requirements.txt
RUN cd /comfyui/models && mkdir facerestore_models && cd facerestore_models && \
 wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth && \
 wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth

RUN cd /comfyui/custom_nodes && git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

EXPOSE 8188

CMD ["/bin/bash", "-c", "source venv/bin/activate && python3 main.py --listen 0.0.0.0"]
