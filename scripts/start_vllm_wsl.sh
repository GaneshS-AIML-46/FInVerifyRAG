#!/bin/bash
# Start vLLM server in WSL Ubuntu-24.04
# Serves Qwen/Qwen2.5-7B-Instruct on port 8000 (OpenAI-compatible API)

set -e

if ! command -v gcc >/dev/null 2>&1; then
  echo "[vLLM] ERROR: gcc not found. Run: sudo apt install -y gcc build-essential"
  exit 1
fi
if [ ! -f /usr/include/python3.12/Python.h ]; then
  echo "[vLLM] ERROR: Python headers missing (needed for Triton). Run:"
  echo "  sudo apt install -y python3-dev build-essential"
  exit 1
fi

source /home/gansi/vllm-env/bin/activate

# FlashInfer JIT needs nvcc; use the CUDA toolkit bundled with pip (nvidia-cu13-*).
VENV_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
export CUDA_HOME="${CUDA_HOME:-$VENV_SITE/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$PATH"
# Avoid FlashInfer sampling JIT (cu13 nvcc vs bundled CCCL headers mismatch on WSL).
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m pip install -q ninja 2>/dev/null || true
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "[vLLM] ERROR: nvcc not found under CUDA_HOME=$CUDA_HOME"
  echo "  Reinstall vLLM env or set CUDA_HOME to a valid CUDA toolkit."
  exit 1
fi
echo "[vLLM] CUDA_HOME=$CUDA_HOME  nvcc=$("$CUDA_HOME/bin/nvcc" --version | head -1)"

echo "[vLLM] Python: $(which python)"
echo "[vLLM] vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
echo "[vLLM] GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Qwen2.5-7B-Instruct in BF16 = ~14GB > 12GB VRAM.
# Use AWQ quantized version (~4.8GB) which fits comfortably.
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${VLLM_MAX_LEN:-8192}"
GPU_MEM="${VLLM_GPU_MEM:-0.90}"

echo "[vLLM] Starting server: model=$MODEL port=$PORT max_model_len=$MAX_LEN gpu_util=$GPU_MEM"
echo "[vLLM] API will be at http://localhost:${PORT}/v1"

# If model doesn't fit FP16, use AWQ quantized version
# Qwen2.5-7B-Instruct in BF16 = ~14GB; need quantization for 12GB VRAM
# --quantization awq only works with AWQ model; use bitsandbytes for on-the-fly 4bit

# --enforce-eager: skip CUDA-graph capture; still needs gcc, python3-dev, and nvcc for JIT kernels.
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --quantization awq \
    --enforce-eager \
    --trust-remote-code \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
    2>&1
