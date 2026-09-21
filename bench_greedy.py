#!/usr/bin/env python3
"""
Greedy speedup search for Qwen3.8-27B (qwen35 hybrid attention) on gfx906.

Stage C showed decode is pinned at ~8.5 tok/s across a 1.45x range of model
sizes and every quant format, so decode is NOT bandwidth or dequant bound.
That is the signature of a fixed per-token cost. With 65 layers, 3/4 of them
linear-attention (many small ops), the prime suspects are command buffer
submission granularity and kernel launch overhead.

Phase 1  env knobs, decode only  (submission granularity, async, fusion)
Phase 2  CLI knobs in one process (flash attn, ubatch, KV quant)
Phase 3  op-level profile via GGML_VK_PERF_LOGGER
Phase 4  speculative decoding with the model's own MTP head
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_bench import (LLAMA, ClockSampler, fmt, run_llama_bench, set_clock,
                       write_csv)

ROOT = Path("/home/ubuntu/Desktop/dirOllamaSetting")
M27 = ROOT / "models27b"
MODEL = M27 / "Qwen3.8-27B-UD-Q3_K_XL.gguf"
DRAFT = M27 / "mtp-Qwen3.8-27B-Q4_0.gguf"
OUT = ROOT / "results27b"
OUT.mkdir(exist_ok=True)

# Phase 1: one knob at a time, decode only. max_nodes_per_submit defaults to 100.
ENV_KNOBS = {
    "base":            {},
    "nodes_1":         {"GGML_VK_MAX_NODES_PER_SUBMIT": "1"},       # control: should be much worse
    "nodes_500":       {"GGML_VK_MAX_NODES_PER_SUBMIT": "500"},
    "nodes_2000":      {"GGML_VK_MAX_NODES_PER_SUBMIT": "2000"},
    "nodes_8192":      {"GGML_VK_MAX_NODES_PER_SUBMIT": "8192"},
    "graphics_queue":  {"GGML_VK_ALLOW_GRAPHICS_QUEUE": "1"},
    "no_async":        {"GGML_VK_DISABLE_ASYNC": "1"},
    "serialize":       {"GGML_VK_SERIALIZE_SUBMISSIONS": "1"},
    "no_fusion":       {"GGML_VK_DISABLE_FUSION": "1"},
    "no_multi_add":    {"GGML_VK_DISABLE_MULTI_ADD": "1"},
    "no_graph_opt":    {"GGML_VK_DISABLE_GRAPH_OPTIMIZE": "1"},
}


def phase1(reps):
    print("\n=== Phase 1: env knobs (decode only, -n 128) ===")
    rows = []
    for name, env in ENV_KNOBS.items():
        r, clk, wall = run_llama_bench(MODEL, env, [], reps=reps, pp=0, tg=128,
                                       timeout=2400)
        rows.append({"phase": "1", "knob": name, "tg_ts": r.get("tg"),
                     "busy": clk["busy_avg"], "wall_s": round(wall, 1),
                     "error": r.get("error")})
        print(f"  {name:16s} tg={fmt(r.get('tg'))}  busy={clk['busy_avg']}%"
              + (f"  ERR={r['error']}" if r.get("error") else ""))
        write_csv(OUT / "greedy_phase1_env.csv", rows)
    ok = [r for r in rows if r["tg_ts"]]
    return max(ok, key=lambda r: r["tg_ts"])["knob"] if ok else "base"


def phase2(best_knob, reps):
    """llama-bench takes comma separated values, so the whole CLI grid runs
    inside a single process and the 13 GB model is loaded only once."""
    print(f"\n=== Phase 2: CLI grid (env={best_knob}, single model load) ===")
    env = dict(os.environ)
    env.update(ENV_KNOBS[best_knob])
    cmd = [str(LLAMA / "llama-bench"), "-m", str(MODEL),
           "-p", "512", "-n", "128", "-r", str(reps), "-ngl", "99",
           "-fa", "0,1",
           "-ub", "128,256,512",
           "-ctk", "f16,q8_0",
           "-o", "json"]
    sampler = ClockSampler()
    sampler.start()
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=9000)
    sampler.stop_flag.set()
    sampler.join(timeout=2)
    (OUT / "greedy_phase2_raw.json").write_text(proc.stdout)
    rows = []
    try:
        for r in json.loads(proc.stdout):
            rows.append({"phase": "2", "fa": r.get("flash_attn"),
                         "ub": r.get("n_ubatch"), "ctk": r.get("type_k"),
                         "ctv": r.get("type_v"),
                         "n_prompt": r.get("n_prompt"), "n_gen": r.get("n_gen"),
                         "ts": r.get("avg_ts")})
    except Exception as e:
        print("  parse failed:", e)
        print(proc.stderr[-1500:])
    write_csv(OUT / "greedy_phase2_cli.csv", rows)
    for r in sorted([x for x in rows if x["n_gen"]], key=lambda x: -(x["ts"] or 0)):
        print(f"  fa={r['fa']} ub={r['ub']} ctk={r['ctk']}  tg={fmt(r['ts'])}")
    print(f"  wall={time.time()-t0:.0f}s")
    return rows


def phase3(best_knob):
    print(f"\n=== Phase 3: op-level profile (GGML_VK_PERF_LOGGER) ===")
    env = dict(os.environ)
    env.update(ENV_KNOBS[best_knob])
    env["GGML_VK_PERF_LOGGER"] = "1"
    cmd = [str(LLAMA / "llama-bench"), "-m", str(MODEL),
           "-p", "0", "-n", "32", "-r", "1", "-ngl", "99", "--no-warmup"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=3600)
    raw = proc.stdout + proc.stderr
    (OUT / "greedy_phase3_perf.txt").write_text(raw)
    print(f"  -> greedy_phase3_perf.txt ({len(raw.splitlines())} lines)")
    return raw


def phase4(best_knob, reps):
    print(f"\n=== Phase 4: speculative decoding with the MTP head ===")
    if not DRAFT.exists():
        print("  draft model missing, skipped")
        return []
    env = dict(os.environ)
    env.update(ENV_KNOBS[best_knob])
    PROMPT = "Explain in detail how HBM2 memory bandwidth limits autoregressive token generation on a GPU without matrix cores."
    # draft-mtp uses the model's own next-token-prediction head (nextn_predict_layers=1)
    runs = [("mtp", "draft-mtp", n) for n in (2, 4, 8)]
    runs += [("ngram", "ngram-simple", 4)]   # needs no draft model, cheap control
    rows = []
    for tag, stype, ndraft in runs:
        cmd = [str(LLAMA / "llama-speculative-simple"),
               "-m", str(MODEL), "-ngl", "99",
               "--spec-type", stype,
               "--spec-draft-n-max", str(ndraft), "--spec-draft-n-min", "1",
               "-p", PROMPT, "-n", "128", "-c", "4096"]
        if stype.startswith("draft-"):
            cmd += ["-md", str(DRAFT), "-ngld", "99"]
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=3600)
        out = proc.stdout + proc.stderr
        (OUT / f"greedy_phase4_{tag}_{ndraft}.txt").write_text(out)
        acc = re.search(r"accept(?:ance)?\s*(?:rate)?\s*[:=]\s*([\d.]+)", out)
        spd = re.findall(r"([\d.]+)\s*tokens per second", out)
        rows.append({"phase": "4", "spec_type": stype, "draft_max": ndraft,
                     "accept": acc.group(1) if acc else None,
                     "tg_ts": spd[-1] if spd else None,
                     "wall_s": round(time.time() - t0, 1),
                     "rc": proc.returncode})
        print(f"  {stype:14s} n_max={ndraft}  accept={acc.group(1) if acc else '?'}  "
              f"tg={spd[-1] if spd else '?'}  rc={proc.returncode}")
    write_csv(OUT / "greedy_phase4_spec.csv", rows)
    return rows


if __name__ == "__main__":
    phases = sys.argv[1] if len(sys.argv) > 1 else "1234"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    set_clock("high")
    best = os.environ.get("BEST_KNOB", "base")
    if "1" in phases:
        best = phase1(reps)
    print(f"\n[best env knob] {best}")
    if "2" in phases:
        phase2(best, reps)
    if "3" in phases:
        phase3(best)
    if "4" in phases:
        phase4(best, reps)
    set_clock("auto")
    print("\n[done]")
