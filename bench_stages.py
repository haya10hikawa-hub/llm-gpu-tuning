#!/usr/bin/env python3
"""
Staged benchmark matrix for gfx906 / Radeon VII.

  A  clock      : DPM state sweep                  (model fixed, env fixed)
  B  env         : Vulkan backend env var sweep    (clock = stage A winner)
  C  quant x env : full cross                      (clock = stage A winner)
  D  op-level    : test-backend-ops MUL_MAT GFLOPS per quant type
  E  graph       : batch size sweep                (best config from C)

Stage C is a full cross because the env vars change *dequant* cost and the
quant type decides how much dequant work there is, so the two are expected
to interact -- that is exactly the effect under test.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_bench import (MODELS, RESULTS, fmt, run_llama_bench, run_op_bench,
                       set_clock, write_csv)

# filled in from `strings libggml-vulkan.so` -- only vars that really exist
ENV_SETS = {}
CLOCKS = [("auto", None), ("high", None), ("manual-mclk0", 0), ("manual-mclk2", 2)]


def load_env_sets(path):
    global ENV_SETS
    ENV_SETS = json.loads(Path(path).read_text())


def models_present():
    # skip draft/projector files: they are not standalone benchmark targets
    skip = ("mtp-", "mmproj", "imatrix")
    return sorted(p for p in MODELS.glob("*.gguf")
                  if not any(s in p.name.lower() for s in skip))


def find_model(tag):
    for m in models_present():
        if tag in m.name:
            return m
    return None


# ---------------------------------------------------------------- stages

def stage_a(model, env, reps):
    print("\n=== Stage A: clock / DPM sweep ===")
    rows = []
    for name, mstate in CLOCKS:
        mode = "manual" if name.startswith("manual") else name
        set_clock(mode, mstate)
        r, clk, wall = run_llama_bench(model, env, [], reps=reps)
        row = {"stage": "A", "clock": name, "model": model.name,
               "env": "default", "pp_ts": r.get("pp"), "tg_ts": r.get("tg"),
               "mclk": clk["mclk_mode"], "sclk": clk["sclk_mode"],
               "busy": clk["busy_avg"], "wall_s": round(wall, 1),
               "error": r.get("error")}
        rows.append(row)
        print(f"  {name:14s} pp={fmt(r.get('pp'))} tg={fmt(r.get('tg'))} "
              f"mclk={clk['mclk_mode']} sclk={clk['sclk_mode']} busy={clk['busy_avg']}%"
              + (f"  ERR={r['error']}" if r.get("error") else ""))
    write_csv(RESULTS / "stage_a_clock.csv", rows)
    return rows


def stage_b(model, clock, reps):
    print(f"\n=== Stage B: env var sweep (clock={clock}) ===")
    set_clock("manual" if clock.startswith("manual") else clock,
              2 if clock == "manual-mclk2" else (0 if clock == "manual-mclk0" else None))
    rows = []
    for name, env in ENV_SETS.items():
        r, clk, wall = run_llama_bench(model, env, [], reps=reps)
        rows.append({"stage": "B", "clock": clock, "model": model.name,
                     "env": name, "pp_ts": r.get("pp"), "tg_ts": r.get("tg"),
                     "mclk": clk["mclk_mode"], "busy": clk["busy_avg"],
                     "wall_s": round(wall, 1), "error": r.get("error")})
        print(f"  {name:34s} pp={fmt(r.get('pp'))} tg={fmt(r.get('tg'))}"
              + (f"  ERR={r['error']}" if r.get("error") else ""))
    write_csv(RESULTS / "stage_b_env.csv", rows)
    return rows


def stage_c(clock, env_names, reps):
    print(f"\n=== Stage C: quant x env cross (clock={clock}) ===")
    set_clock("manual" if clock.startswith("manual") else clock,
              2 if clock == "manual-mclk2" else (0 if clock == "manual-mclk0" else None))
    rows = []
    for m in models_present():
        for name in env_names:
            env = ENV_SETS[name]
            r, clk, wall = run_llama_bench(m, env, [], reps=reps)
            rows.append({"stage": "C", "clock": clock, "model": m.name,
                         "env": name, "pp_ts": r.get("pp"), "tg_ts": r.get("tg"),
                         "size_b": r.get("model_size"),
                         "mclk": clk["mclk_mode"], "busy": clk["busy_avg"],
                         "wall_s": round(wall, 1), "error": r.get("error")})
            print(f"  {m.name:34s} {name:22s} pp={fmt(r.get('pp'))} tg={fmt(r.get('tg'))}"
                  + (f"  ERR={r['error']}" if r.get("error") else ""))
            write_csv(RESULTS / "stage_c_quant_env.csv", rows)
    return rows


def stage_d(clock, env_names):
    print(f"\n=== Stage D: op-level MUL_MAT GFLOPS (clock={clock}) ===")
    set_clock("manual" if clock.startswith("manual") else clock,
              2 if clock == "manual-mclk2" else (0 if clock == "manual-mclk0" else None))
    for name in env_names:
        out, clk = run_op_bench(ENV_SETS[name])
        p = RESULTS / f"stage_d_mulmat_{name}.txt"
        p.write_text(out)
        print(f"  {name:22s} -> {p.name} ({len(out.splitlines())} lines, mclk={clk['mclk_mode']})")


def stage_e(model, clock, env_name, reps):
    print(f"\n=== Stage E: batch sweep (clock={clock}, env={env_name}) ===")
    set_clock("manual" if clock.startswith("manual") else clock,
              2 if clock == "manual-mclk2" else (0 if clock == "manual-mclk0" else None))
    rows = []
    for bs in [1, 2, 4, 8, 16]:
        r, clk, wall = run_llama_bench(model, ENV_SETS[env_name],
                                       ["-pg", f"0,{128}"] if False else [],
                                       reps=reps, pp=512, tg=128)
        rows.append({"stage": "E", "batch": bs, "model": model.name,
                     "pp_ts": r.get("pp"), "tg_ts": r.get("tg"),
                     "error": r.get("error")})
    write_csv(RESULTS / "stage_e_batch.csv", rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-sets", default=str(RESULTS / "env_sets.json"))
    ap.add_argument("--stages", default="ABCD")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--base-model", default="UD-Q3_K_XL")
    ap.add_argument("--clock", default=None, help="skip stage A, force this clock")
    ap.add_argument("--best-env", default=None, help="skip stage B, force this env set")
    ap.add_argument("--cross-envs", default=None,
                    help="comma separated env set names for stage C")
    args = ap.parse_args()

    load_env_sets(args.env_sets)
    base = find_model(args.base_model)
    if base is None:
        sys.exit(f"base model matching {args.base_model!r} not found in {MODELS}")
    print(f"base model : {base.name}")
    print(f"env sets   : {list(ENV_SETS)}")
    print(f"models     : {[m.name for m in models_present()]}")

    best_clock = args.clock
    if "A" in args.stages:
        rows = stage_a(base, ENV_SETS["default"], args.reps)
        ok = [r for r in rows if r["tg_ts"]]
        if ok:
            best_clock = max(ok, key=lambda r: r["tg_ts"])["clock"]
    best_clock = best_clock or "high"
    print(f"\n[clock chosen] {best_clock}")

    best_env = args.best_env or "default"
    if "B" in args.stages:
        rows = stage_b(base, best_clock, args.reps)
        ok = [r for r in rows if r["tg_ts"]]
        if ok:
            best_env = max(ok, key=lambda r: r["tg_ts"])["env"]
    print(f"[env chosen  ] {best_env}")

    # quant x env cross: keep the env axis to the factors that change dequant
    # cost, otherwise the cross explodes without telling us anything new.
    if args.cross_envs:
        cross = [e.strip() for e in args.cross_envs.split(",") if e.strip()]
    else:
        cross = sorted({"default", "user_original", "no_int_dot", best_env})
    if "C" in args.stages:
        stage_c(best_clock, cross, args.reps)
    if "D" in args.stages:
        stage_d(best_clock, sorted({"default", best_env}))

    set_clock("auto")
    print("\n[done] clocks restored to auto")


if __name__ == "__main__":
    main()
