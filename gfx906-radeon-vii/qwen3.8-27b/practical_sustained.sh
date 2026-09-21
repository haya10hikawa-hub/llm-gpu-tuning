#!/usr/bin/env bash
# Sustained load test: keep llama-server resident, hammer it with requests and
# track clocks / temperature / power / VRAM. Covers thermal throttling,
# allocator drift and stability in one run.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
B=${LLAMA_BIN:-$REPO/work/llama.cpp/build/bin}
M=$HERE/models/Qwen3.8-27B-UD-Q4_K_S.gguf
C=/sys/class/drm/card1/device
OUT=$HERE/results/sustained.csv
LOG=${TMPDIR:-/tmp}
DURATION=${1:-1500}   # seconds
PORT=11436

$REPO/gfx906-radeon-vii/gpuclk.sh high >/dev/null 2>&1
hw(){ ls $C/hwmon/hwmon*/$1 2>/dev/null | head -1; }

GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 nohup $B/llama-server -m $M -ngl 99 -c 8192 \
  --host 127.0.0.1 --port $PORT --parallel 1 > $LOG/srv.log 2>&1 &
SRV=$!
echo "server pid=$SRV, waiting for load..."
for i in $(seq 1 120); do
  curl -s -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3
done
curl -s -m 5 http://127.0.0.1:$PORT/health | head -c 200; echo

echo "t_sec,iter,decode_ts,prefill_ts,pred_n,sclk_mhz,mclk_mhz,edge_c,junction_c,mem_c,power_w,vram_mb,busy" > $OUT
P="Write a detailed technical explanation of how memory bandwidth limits token generation on GPUs without matrix cores. Cover the roofline model, dequantization cost, and batching."
T0=$(date +%s); i=0
while [ $(( $(date +%s) - T0 )) -lt $DURATION ]; do
  i=$((i+1))
  R=$(curl -s -m 300 http://127.0.0.1:$PORT/completion \
       -H 'Content-Type: application/json' \
       -d "{\"prompt\":\"$P\",\"n_predict\":128,\"cache_prompt\":false,\"temperature\":0}")
  dt=$(echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);t=d.get('timings',{});print(f\"{t.get('predicted_per_second',0):.2f},{t.get('prompt_per_second',0):.2f},{t.get('predicted_n',0)}\")" 2>/dev/null || echo "0,0,0")
  s=$(grep '\*' $C/pp_dpm_sclk | grep -oE '[0-9]+Mhz' | grep -oE '[0-9]+'); m=$(grep '\*' $C/pp_dpm_mclk | grep -oE '[0-9]+Mhz' | grep -oE '[0-9]+')
  e=$(cat "$(hw temp1_input)" 2>/dev/null); j=$(cat "$(hw temp2_input)" 2>/dev/null); mm=$(cat "$(hw temp3_input)" 2>/dev/null)
  p=$(cat "$(hw power1_input)" 2>/dev/null); v=$(cat $C/mem_info_vram_used); bz=$(cat $C/gpu_busy_percent)
  echo "$(( $(date +%s) - T0 )),$i,$dt,${s:-0},${m:-0},$((${e:-0}/1000)),$((${j:-0}/1000)),$((${mm:-0}/1000)),$(( ${p:-0}/1000000 )),$((v/1024/1024)),$bz" >> $OUT
  printf "  t=%-5s iter=%-3s tg=%-7s sclk=%-5s junc=%-4s mem=%-4s %sW\n" \
     "$(( $(date +%s) - T0 ))" "$i" "$(echo $dt|cut -d, -f1)" "${s:-?}" "$((${j:-0}/1000))" "$((${mm:-0}/1000))" "$(( ${p:-0}/1000000 ))"
done
kill $SRV 2>/dev/null; sleep 3
echo "=== done -> $OUT ==="
