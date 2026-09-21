#!/usr/bin/env bash
# Perplexity across the 27B quant series. Speed work so far never checked
# quality, so this measures the degradation curve on a common corpus.
set -u
ROOT=/home/ubuntu/Desktop/dirOllamaSetting
B=$ROOT/work/llama.cpp/build/bin
CORPUS=$ROOT/corpus/wiki.test.raw
OUT=$ROOT/results27b/quality_ppl.csv
LOG=/tmp/claude-1000/-home-ubuntu-Desktop-dirOllamaSetting/60c74b66-1de7-445b-8ab5-15348a1f5583/scratchpad
CHUNKS=${1:-40}

$ROOT/gpuclk.sh high >/dev/null 2>&1
echo "model,size_gb,ppl,ppl_err,status" > $OUT

for name in Qwen3.8-27B-UD-Q2_K_XL Qwen3.8-27B-UD-IQ3_XXS Qwen3.8-27B-UD-Q3_K_XL \
            Qwen3.8-27B-UD-IQ4_XS Qwen3.8-27B-UD-Q4_K_S; do
  f=$ROOT/models27b/$name.gguf
  [ -s "$f" ] || { echo "$name,,,,MISSING" >> $OUT; continue; }
  sz=$(stat -c%s "$f")

  busy=$(cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || echo 0)
  if [ "$busy" -gt 20 ]; then echo "$name,,,,GPU_BUSY" >> $OUT; continue; fi

  GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 timeout 3600 \
    $B/llama-perplexity -m "$f" -f $CORPUS -ngl 99 -c 512 --chunks $CHUNKS \
    > $LOG/ppl.log 2>&1
  if grep -qa "no usable GPU" $LOG/ppl.log; then
    echo "$name,$(python3 -c "print(f'{$sz/1e9:.2f}')"),,,GPUFAIL" >> $OUT
    printf "  %-30s GPUFAIL\n" "$name"; continue
  fi
  line=$(sed 's/\x1b\[[0-9;]*m//g' $LOG/ppl.log | grep -a "Final estimate" | tail -1)
  ppl=$(echo "$line" | grep -oE "PPL = [0-9.]+" | grep -oE "[0-9.]+")
  err=$(echo "$line" | grep -oE "\+/- [0-9.]+" | grep -oE "[0-9.]+")
  st="OK"; [ -z "$ppl" ] && st="PARSEFAIL"
  python3 -c "print(f'{\"$name\"},{$sz/1e9:.2f},{\"$ppl\"},{\"$err\"},{\"$st\"}')" >> $OUT
  printf "  %-30s %6.2f GB  PPL=%-9s +/- %-8s %s\n" "$name" "$(python3 -c "print($sz/1e9)")" "${ppl:-?}" "${err:-?}" "$st"
done

echo "=== done -> $OUT ==="
$ROOT/gpuclk.sh auto >/dev/null 2>&1
