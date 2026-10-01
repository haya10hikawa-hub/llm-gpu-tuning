#!/usr/bin/env bash
# Bonsai 2 27B ternary GGUFs. Both need the PrismML llama.cpp fork
# (https://github.com/PrismML-Eng/llama.cpp); upstream llama.cpp cannot load them.
set -u
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models"
REPO=prism-ml/Ternary-Bonsai-2-27B-gguf
mkdir -p "$DEST"

FILES=(
  Ternary-Bonsai-2-27B-PQ2_0.gguf    # 7.21 GB  measured format
  Ternary-Bonsai-2-27B-PTQ1_0.gguf   # 5.95 GB  smaller variant, not measured yet
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
df -h "$DEST" | tail -1
