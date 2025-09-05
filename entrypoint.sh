#!/usr/bin/env bash
set -euo pipefail

NV="${NV_MOUNT:-/workspace}"
COMFY_DIR="$NV/ComfyUI"
VENV_DIR="$NV/venv"
CUSTOM_NODES_DIR="$COMFY_DIR/custom_nodes"
MODELS_DIR="$COMFY_DIR/models"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$NV/cache}"
export HF_HOME="${HF_HOME:-$NV/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$NV/cache/pip}"

mkdir -p "$NV" "$XDG_CACHE_HOME" "$HF_HOME" "$PIP_CACHE_DIR"

# 0) проверим, что NV смонтирован и доступен на запись
if ! mkdir -p "$NV/.rw-test" 2>/dev/null; then
  echo "[!] Network Volume не смонтирован в $NV. Будет использоваться эфемерное /comfyui."
  NV="/comfyui-ephemeral"
  COMFY_DIR="$NV/ComfyUI"
  VENV_DIR="$NV/venv"
  CUSTOM_NODES_DIR="$COMFY_DIR/custom_nodes"
  MODELS_DIR="$COMFY_DIR/models"
  mkdir -p "$NV"
else
  rm -rf "$NV/.rw-test" || true
fi

echo "[i] NV: $NV"
echo "[i] Comfy: $COMFY_DIR"
echo "[i] Caches: XDG_CACHE_HOME=$XDG_CACHE_HOME, HF_HOME=$HF_HOME, PIP_CACHE_DIR=$PIP_CACHE_DIR"

# 1) ComfyUI: clone или обновить
if [ ! -d "$COMFY_DIR/.git" ]; then
  echo "[i] Cloning ComfyUI into NV…"
  git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
  git -C "$COMFY_DIR" submodule update --init --recursive || true
# else
#   echo "[i] Updating ComfyUI…"
#   git -C "$COMFY_DIR" fetch --depth=1 origin || true
#   git -C "$COMFY_DIR" reset --hard origin/master || git -C "$COMFY_DIR" pull --ff-only || true
#   git -C "$COMFY_DIR" submodule update --init --recursive || true
fi

# 2) venv в NV
if [ ! -d "$VENV_DIR" ]; then
  echo "[i] Creating venv…"
  python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# 3) базовые зависимости (CUDA 12.1)
pip install --no-cache-dir \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4) зависимости ComfyUI
pip install --no-cache-dir -r "$COMFY_DIR/requirements.txt"

# 5) твои кастом-ноды (ставим только если их ещё нет)
mkdir -p "$CUSTOM_NODES_DIR" "$MODELS_DIR"

# ComfyUI-Manager
if [ ! -d "$CUSTOM_NODES_DIR/comfyui-manager/.git" ] && [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-Manager/.git" ]; then
  git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Manager.git "$CUSTOM_NODES_DIR/ComfyUI-Manager" || true
fi

# ReActor
if [ ! -d "$CUSTOM_NODES_DIR/comfyui-reactor-node/.git" ]; then
  git clone --depth=1 https://codeberg.org/Gourieff/comfyui-reactor-node.git "$CUSTOM_NODES_DIR/comfyui-reactor-node" || true
  pip install --no-cache-dir opencv-python onnxruntime-gpu insightface
  python "$CUSTOM_NODES_DIR/comfyui-reactor-node/install.py" || true
  mkdir -p "$MODELS_DIR/ultralytics/bbox"
  if [ ! -f "$MODELS_DIR/ultralytics/bbox/face_yolov8m.pt" ]; then
    wget -q -O "$MODELS_DIR/ultralytics/bbox/face_yolov8m.pt" \
      https://huggingface.co/datasets/Gourieff/ReActor/raw/main/models/detection/bbox/face_yolov8m.pt || true
  fi
fi

# FaceRestore CF
if [ ! -d "$CUSTOM_NODES_DIR/facerestore_cf/.git" ]; then
  git clone --depth=1 https://github.com/mav-rik/facerestore_cf.git "$CUSTOM_NODES_DIR/facerestore_cf" || true
  pip install --no-cache-dir -r "$CUSTOM_NODES_DIR/facerestore_cf/requirements.txt" || true
  mkdir -p "$MODELS_DIR/facerestore_models"
  [ -f "$MODELS_DIR/facerestore_models/GFPGANv1.4.pth" ] || \
    wget -q -P "$MODELS_DIR/facerestore_models" https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth || true
  [ -f "$MODELS_DIR/facerestore_models/codeformer.pth" ] || \
    wget -q -P "$MODELS_DIR/facerestore_models" https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth || true
fi

# IPAdapter+
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI_IPAdapter_plus/.git" ]; then
  git clone --depth=1 https://github.com/cubiq/ComfyUI_IPAdapter_plus.git "$CUSTOM_NODES_DIR/ComfyUI_IPAdapter_plus" || true
fi

# Extra API
if [ ! -d "$CUSTOM_NODES_DIR/comfyui_extra_api/.git" ]; then
  git clone --depth=1 https://github.com/injet-zhou/comfyui_extra_api.git "$CUSTOM_NODES_DIR/comfyui_extra_api" || true
  pip install --no-cache-dir -r "$CUSTOM_NODES_DIR/comfyui_extra_api/requirements.txt" || true
fi

# GGUF
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-GGUF/.git" ]; then
  git clone --depth=1 https://github.com/city96/ComfyUI-GGUF "$CUSTOM_NODES_DIR/ComfyUI-GGUF" || true
  pip install --no-cache-dir --upgrade gguf || true
fi

if [ ! -e "/comfyui" ]; then
  ln -s "$COMFY_DIR" /comfyui
fi

# MODELS

# gguf
[ -f "$MODELS_DIR/unet/model.gguf" ] || (mkdir -p "$MODELS_DIR/unet" && cd "$MODELS_DIR/unet" && wget "https://civitai.com/api/download/models/2125565?token=$CIVITAI_TOKEN" -O model.gguf)

# clip (t5xxl_fp8)
[ -f "$MODELS_DIR/clip/t5xxl_fp8_e4m3fn_scaled.safetensors" ] || (
  cd "$MODELS_DIR/clip" && \
  wget "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors"
)

# clip (vit-large-patch14)
[ -f "$MODELS_DIR/clip/model.safetensors" ] || (
  cd "$MODELS_DIR/clip" && \
  wget "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors"
)

# vae (нужен токен HF)
[ -f "$MODELS_DIR/vae/diffusion_pytorch_model.safetensors" ] || (
  cd "$MODELS_DIR/vae" && \
  wget --header="Authorization: Bearer $HUGGINGFACE_TOKEN" "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/diffusion_pytorch_model.safetensors"
)

# lora
[ -f "$MODELS_DIR/loras/lora.safetensors" ] || (
  cd "$MODELS_DIR/loras" && \
  wget "https://huggingface.co/XLabs-AI/flux-RealismLora/resolve/main/lora.safetensors"
)

cd "$COMFY_DIR"
echo "[i] Starting ComfyUI on 0.0.0.0:${COMFY_PORT:-8188}"
exec python main.py --listen 0.0.0.0 --port "${COMFY_PORT:-8188}"
