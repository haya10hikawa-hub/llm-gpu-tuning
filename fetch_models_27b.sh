#!/usr/bin/env bash
# Qwen3.8-27B GGUF variants that fit fully in 16 GB VRAM.
# Q4_K_M / Q5 / Q6 / Q8_0 are deliberately excluded: they exceed VRAM and would
# force partial CPU offload, which makes the GPU comparison meaningless.
set -u
DEST=/home/ubuntu/Desktop/dirOllamaSetting/models27b
REPO=unsloth/Qwen3.8-27B-GGUF
mkdir -p "$DEST"

FILES=(
  Qwen3.8-27B-UD-Q2_K_XL.gguf    #  9.83 GB  K-quant, most headroom
  Qwen3.8-27B-UD-IQ3_XXS.gguf    # 10.93 GB  I-quant, user's comparison point
  Qwen3.8-27B-UD-Q3_K_XL.gguf    # 13.15 GB  K-quant, USER'S BASELINE
  Qwen3.8-27B-UD-IQ4_XS.gguf     # 14.25 GB  I-quant 4-bit
  Qwen3.8-27B-UD-Q4_K_S.gguf     # 15.36 GB  K-quant 4-bit, VRAM edge probe
  MTP/mtp-Qwen3.8-27B-Q4_0.gguf  #  1.37 GB  multi-token-prediction draft head
)

for f in "${FILES[@]}"; do
  out="$DEST/$(basename "$f")"
  if [ -s "$out" ]; then echo "[skip] $f"; continue; fi
  echo "[get ] $f"
  curl -fL --no-progress-meter --retry 5 --retry-delay 3 -C - \
    -o "$out" "https://huggingface.co/$REPO/resolve/main/$f" \
    || { echo "[FAIL] $f"; rm -f "$out"; }
done

echo "=== done ==="
ls -la "$DEST"
df -h /home | tail -1
