#!/usr/bin/env bash
# Download the Qwen3-8B GGUF variants used by the benchmark matrix.
set -u
DEST=/home/ubuntu/Desktop/dirOllamaSetting/models
REPO=unsloth/Qwen3-8B-GGUF
mkdir -p "$DEST"

FILES=(
  Qwen3-8B-UD-Q3_K_XL.gguf   # baseline (user's current setup)
  Qwen3-8B-UD-IQ3_XXS.gguf   # LUT quant at comparable bpw
  Qwen3-8B-IQ4_XS.gguf       # predicted sweet spot
  Qwen3-8B-Q8_0.gguf         # "dequant is nearly free" hypothesis test
  Qwen3-8B-Q4_K_M.gguf       # K-quant control
)

for f in "${FILES[@]}"; do
  out="$DEST/$f"
  if [ -s "$out" ]; then echo "[skip] $f"; continue; fi
  echo "[get ] $f"
  curl -fL --retry 5 --retry-delay 3 -C - \
    -o "$out" "https://huggingface.co/$REPO/resolve/main/$f" \
    || { echo "[FAIL] $f"; rm -f "$out"; }
done

echo "=== done ==="
ls -la "$DEST"
df -h /home | tail -1
