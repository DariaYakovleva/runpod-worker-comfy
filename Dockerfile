FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv git wget curl ca-certificates \
    libglib2.0-0 libgl1 ffmpeg && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

ENV NV_MOUNT=/workspace
ENV COMFY_PORT=8188

ENV XDG_CACHE_HOME=${NV_MOUNT}/cache
ENV HF_HOME=${NV_MOUNT}/huggingface
ENV PIP_CACHE_DIR=${NV_MOUNT}/cache/pip

WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8188
ENTRYPOINT ["/app/entrypoint.sh"]
