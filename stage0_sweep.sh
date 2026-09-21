#!/usr/bin/env bash
# Stage 0: re-run the env sweep that the earlier (fragmented / llama-bench) runs invalidated.
# One GPU job at a time, GPU use verified per run, control interleaved to catch drift.
set -u
ROOT=/home/ubuntu/Desktop/dirOllamaSetting
B=$ROOT/work/llama.cpp/build/bin
M=$ROOT/models27b/Qwen3.8-27B-UD-Q4_K_S.gguf
OUT=$ROOT/results27b/stage0_env.csv
LOG=/tmp/claude-1000/-home-ubuntu-Desktop-dirOllamaSetting/60c74b66-1de7-445b-8ab5-15348a1f5583/scratchpad
PROMPT=$LOG/long.txt

# base config applied to every run
BASE_ENV=(GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1)

$ROOT/gpuclk.sh high >/dev/null 2>&1
[ -s "$PROMPT" ] || python3 -c "
t='The Radeon VII is a Vega 20 GPU with 60 compute units, 16 GB of HBM2 memory on a 4096-bit bus, and no matrix cores. Autoregressive decoding reads every weight once per token, so throughput is bounded by memory bandwidth unless dequantization becomes the limit. '
open('$PROMPT','w').write(t*14)"

echo "name,prefill_ts,decode_ts,pp_tokens,status" > $OUT

run() {
  name="$1"; shift
  # refuse to start if the GPU is busy
  busy=$(cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || echo 0)
  if [ "$busy" -gt 20 ]; then echo "$name,,,,GPU_BUSY" >> $OUT; printf "  %-36s GPU_BUSY\n" "$name"; return; fi

  timeout 900 env "${BASE_ENV[@]}" "$@" \
    $B/llama-completion -m $M -ngl 99 -c 4096 -n 128 --seed 1 -f $PROMPT > $LOG/s0.log 2>&1
  rc=$?
  if grep -qa "no usable GPU" $LOG/s0.log; then
    echo "$name,,,,GPUFAIL" >> $OUT; printf "  %-36s GPUFAIL\n" "$name"; return
  fi
  L=$(sed 's/\x1b\[[0-9;]*m//g' $LOG/s0.log)
  pp=$(echo "$L" | grep -a "prompt eval time" | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+")
  tg=$(echo "$L" | grep -a "  eval time" | grep -v prompt | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+")
  nt=$(echo "$L" | grep -a "prompt eval time" | grep -oE "/ +[0-9]+ tokens" | grep -oE "[0-9]+")
  st="OK"; [ -z "$tg" ] && st="PARSEFAIL(rc=$rc)"
  echo "$name,${pp:-},${tg:-},${nt:-},$st" >> $OUT
  printf "  %-36s pp=%-9s tg=%-9s %s\n" "$name" "${pp:-?}" "${tg:-?}" "$st"
}

echo "=== Stage 0: env sweep (base = DISABLE_HOST_VISIBLE_VIDMEM=1) ==="
run "base(control-1)"                   X=1
run "FORCE_MMVQ"                        GGML_VK_FORCE_MMVQ=1
run "DISABLE_MMVQ"                      GGML_VK_DISABLE_MMVQ=1
run "base(control-2)"                   X=1
run "DISABLE_INTEGER_DOT_PRODUCT"       GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1
run "DISABLE_DOT2"                      GGML_VK_DISABLE_DOT2=1
run "base(control-3)"                   X=1
run "DISABLE_F16"                       GGML_VK_DISABLE_F16=1
run "ALLOW_GRAPHICS_QUEUE"              GGML_VK_ALLOW_GRAPHICS_QUEUE=1
run "base(control-4)"                   X=1
run "DISABLE_FUSION"                    GGML_VK_DISABLE_FUSION=1
run "DISABLE_GRAPH_OPTIMIZE"            GGML_VK_DISABLE_GRAPH_OPTIMIZE=1
run "base(control-5)"                   X=1
run "DISABLE_MULTI_ADD"                 GGML_VK_DISABLE_MULTI_ADD=1
run "MAX_NODES_PER_SUBMIT=2000"         GGML_VK_MAX_NODES_PER_SUBMIT=2000
run "DISABLE_ASYNC"                     GGML_VK_DISABLE_ASYNC=1
run "base(control-6)"                   X=1

echo "=== done -> $OUT ==="
$ROOT/gpuclk.sh auto >/dev/null 2>&1
