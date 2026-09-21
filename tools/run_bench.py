#!/usr/bin/env python3
"""
gfx906 (Radeon VII) benchmark matrix runner.

Staged design: each stage fixes the winner of the previous stage, so the
number of runs stays bounded, and dimensions that plausibly interact
(env x quant) are run as a full cross instead of one-at-a-time.

Every run records the mclk/sclk DPM state sampled *during* the run, because
on Vega 20 a compute-only load often leaves mclk parked at DPM 0 (350 MHz),
which silently caps HBM2 at 358 GB/s instead of 1024 GB/s.
"""

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

# repo root is the parent of tools/. Override with LLM_TUNING_ROOT.
ROOT = Path(os.environ.get("LLM_TUNING_ROOT", Path(__file__).resolve().parents[1]))
LLAMA = Path(os.environ.get("LLAMA_BIN", ROOT / "work/llama.cpp/build/bin"))
MODELS = Path(os.environ.get("BENCH_MODELS", ROOT / "models"))
RESULTS = Path(os.environ.get("BENCH_RESULTS", ROOT / "results"))
CARD = Path("/sys/class/drm/card1/device")

RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------- clocks

def set_clock(mode, mclk_state=None):
    """mode: auto | high | manual"""
    lvl = CARD / "power_dpm_force_performance_level"
    subprocess.run(["sudo", "tee", str(lvl)], input=mode.encode(),
                   stdout=subprocess.DEVNULL, check=False)
    if mode == "manual" and mclk_state is not None:
        subprocess.run(["sudo", "tee", str(CARD / "pp_dpm_mclk")],
                       input=str(mclk_state).encode(),
                       stdout=subprocess.DEVNULL, check=False)
    time.sleep(1.0)


def _read_star(path):
    try:
        for line in path.read_text().splitlines():
            if "*" in line:
                return line.split(":")[1].replace("*", "").strip()
    except Exception:
        pass
    return "?"


class ClockSampler(threading.Thread):
    """Samples DPM state while a benchmark runs."""

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.mclk, self.sclk, self.busy = [], [], []

    def run(self):
        while not self.stop_flag.is_set():
            self.mclk.append(_read_star(CARD / "pp_dpm_mclk"))
            self.sclk.append(_read_star(CARD / "pp_dpm_sclk"))
            try:
                self.busy.append(int((CARD / "gpu_busy_percent").read_text().strip()))
            except Exception:
                pass
            time.sleep(self.interval)

    def summary(self):
        def mode(xs):
            return max(set(xs), key=xs.count) if xs else "?"
        busy = [b for b in self.busy if b > 5]
        return {
            "mclk_mode": mode(self.mclk),
            "mclk_max": max(self.mclk, default="?", key=_mhz),
            "sclk_mode": mode(self.sclk),
            "busy_avg": round(sum(busy) / len(busy), 1) if busy else 0.0,
        }


def _mhz(s):
    try:
        return int(str(s).replace("Mhz", "").replace("MHz", "").strip())
    except Exception:
        return -1


# ---------------------------------------------------------------- runners

def run_llama_bench(model, env_extra, extra_args, reps=3, pp=512, tg=128,
                    timeout=1800):
    cmd = [str(LLAMA / "llama-bench"), "-m", str(model),
           "-p", str(pp), "-n", str(tg), "-r", str(reps),
           "-ngl", "99", "-o", "json"] + extra_args
    env = dict(os.environ)
    env.update(env_extra)

    sampler = ClockSampler()
    sampler.start()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        sampler.stop_flag.set()
        return {"error": "timeout"}, sampler.summary(), 0.0
    finally:
        sampler.stop_flag.set()
        sampler.join(timeout=2)
    wall = time.time() - t0

    out = {"error": None, "pp": None, "tg": None, "stderr_tail": ""}
    try:
        rows = json.loads(proc.stdout)
        for r in rows:
            ts = float(r["avg_ts"])
            if r.get("n_prompt", 0) > 0:
                out["pp"] = ts
            elif r.get("n_gen", 0) > 0:
                out["tg"] = ts
        out["model_size"] = rows[0].get("model_size") if rows else None
    except Exception as e:
        out["error"] = f"parse: {e} rc={proc.returncode}"
        out["stderr_tail"] = proc.stderr[-800:]
    return out, sampler.summary(), wall


def run_op_bench(env_extra, op="MUL_MAT", timeout=3600):
    """test-backend-ops perf -- per-quant GFLOPS, needs no model."""
    cmd = [str(LLAMA / "test-backend-ops"), "perf", "-o", op, "-b", "Vulkan0"]
    env = dict(os.environ)
    env.update(env_extra)
    sampler = ClockSampler()
    sampler.start()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        sampler.stop_flag.set()
        return "", sampler.summary()
    finally:
        sampler.stop_flag.set()
        sampler.join(timeout=2)
    return proc.stdout, sampler.summary()


# ---------------------------------------------------------------- output

def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"
