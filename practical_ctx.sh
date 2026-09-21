#!/usr/bin/env bash
# 文脈長スケーリング: 深さを変えて decode/prefill を測る。
# 65層中 49層が線形アテンション (状態固定) なので、通常の Transformer より
# 劣化が緩やかなはず、という仮説の検証。
set -u
ROOT=/home/ubuntu/Desktop/dirOllamaSetting
B=$ROOT/work/llama.cpp/build/bin
M=$ROOT/models27b/Qwen3.8-27B-UD-Q3_K_XL.gguf
C=/sys/class/drm/card1/device
LOG=/tmp/claude-1000/-home-ubuntu-Desktop-dirOllamaSetting/60c74b66-1de7-445b-8ab5-15348a1f5583/scratchpad
OUT=$ROOT/results27b/ctx_scaling.csv
PORT=11437
CTK=${1:-f16}

$ROOT/gpuclk.sh high >/dev/null 2>&1
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 nohup $B/llama-server -m $M -ngl 99 -c 49152 \
  -ctk $CTK -ctv $CTK --host 127.0.0.1 --port $PORT --parallel 1 > $LOG/srv2.log 2>&1 &
SRV=$!
for i in $(seq 1 150); do curl -s -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
echo "server up (ctk=$CTK) VRAM=$(( $(cat $C/mem_info_vram_used)/1024/1024 ))MB"

echo "ctk,target_tok,prompt_n,prefill_ts,decode_ts,vram_mb" > $OUT
for NTOK in 512 4096 16384 32768; do
  CHARS=$(( NTOK * 4 ))
  python3 -c "
import json,sys
t='The Radeon VII is a Vega 20 GPU with 60 compute units and 16 GB of HBM2 memory. Autoregressive decoding reads every weight once per token. '
p=(t*4000)[:$CHARS]
print(json.dumps({'prompt':p,'n_predict':64,'cache_prompt':False,'temperature':0.7}))" > $LOG/req.json
  R=$(curl -s -m 900 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' --data-binary @$LOG/req.json)
  echo "$R" | python3 -c "
import sys,json
d=json.load(sys.stdin); t=d.get('timings',{})
import subprocess
v=int(open('/sys/class/drm/card1/device/mem_info_vram_used').read())//1024//1024
print(f\"$CTK,$NTOK,{t.get('prompt_n',0)},{t.get('prompt_per_second',0):.2f},{t.get('predicted_per_second',0):.2f},{v}\")" >> $OUT 2>/dev/null \
   || echo "$CTK,$NTOK,,,,ERR" >> $OUT
  tail -1 $OUT
done
kill $SRV 2>/dev/null; sleep 3
