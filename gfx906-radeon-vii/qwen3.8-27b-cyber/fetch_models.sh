#!/usr/bin/env bash
# Qwen3.8-27B-Uncensored-Cyber IQ4_XS with the MTP head grafted in (one file; run it without
# --spec-type for the no-MTP baseline). Pinned revision, resumable, sha256-checked.
# Partial downloads stay next to the target (never /tmp: it may be RAM-backed).
set -u
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models"
REPO=cyjin-yl/Qwen3.8-27B-Uncensored-Cyber-agentic-imatrix-GGUF
REV=d82fb040934e0a491600a49477114d005accf59f
FILE=Qwen3.8-27B-Uncensored-Cyber-IQ4_XS-imatrix-fromq8-plus-mtp.gguf   # 16,517,175,520 bytes (15.38 GiB)
SHA256=da6a418f30a7e6c6669b74179f6d533ca06016e02eba298d2b713a2900d7a1ba
mkdir -p "$DEST"
export TMPDIR="$DEST"

avail_gb=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc 0-9)
if [ "$avail_gb" -lt 16 ]; then echo "[STOP] only ${avail_gb} GB free under $DEST (need 16)"; exit 1; fi

out="$DEST/$FILE"
if [ -s "$out" ]; then echo "[skip] $FILE"; else
  echo "[get ] $FILE @ $REV"
  curl -fL --no-progress-meter --retry 20 --retry-delay 5 --retry-all-errors -C - \
    -o "$out.part" "https://huggingface.co/$REPO/resolve/$REV/$FILE" || { echo "[FAIL] download"; exit 1; }
  got=$(sha256sum "$out.part" | cut -d' ' -f1)
  if [ "$got" != "$SHA256" ]; then echo "[FAIL] sha256 $got != $SHA256 (removing, rerun to fetch again)"; rm -f "$out.part"; exit 1; fi
  mv "$out.part" "$out"
fi

echo "=== done ==="
ls -la "$DEST"
df -h "$DEST" | tail -1
