"""S3: slot arbitration + throughput.

v2. v1 used tokens/wall-clock, which folds in prefill, HTTP and queue wait, and
is NOT what the repo's 1.48x figure measures -- that is the S_TG (generation)
column. Single stream read 17.9 t/s on the wall-clock metric against a known
~25 t/s decode, which is how the error surfaced. Both metrics are reported here,
labelled, and only the decode metric is compared against the 1.48x ceiling.
"""
import os, statistics as st, subprocess, sys, threading, time
from local_pool import Degraded, LocalPool

BASE = os.environ.get("LLM_API", "http://127.0.0.1:8080")
N_PRED = int(os.environ.get("N_PRED", "128"))
TOPICS = ["a red bicycle", "the tide at dawn", "a copper kettle", "an empty platform",
          "winter birches", "a paper lantern", "the last ferry", "a cracked bell"]
prompt = lambda i: f"Write exactly three plain sentences about {TOPICS[i%len(TOPICS)]}. No preamble."

def llama_procs():
    out = subprocess.run(["bash","-c","ls /proc/*/exe 2>/dev/null | xargs -r -I{} readlink {} 2>/dev/null | grep -ci llama"],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)

def run_batch(pool, n, watch=False):
    recs, errs, peak, stop = [], [], [0], threading.Event()
    def watcher():
        while not stop.is_set():
            b = pool.slots_busy()
            if b is not None and b > peak[0]: peak[0] = b
            time.sleep(0.05)
    def worker(i):
        try:
            t_end_ref = {}
            r = pool.complete(prompt(i), n_predict=N_PRED)
            recs.append({"n": r["n"], "tps": r["tps"], "end": time.perf_counter(),
                         "dec_ms": r["n"] / r["tps"] * 1000.0 if r["tps"] else 0.0})
        except Exception as e: errs.append(repr(e))
    w = threading.Thread(target=watcher, daemon=True)
    if watch: w.start()
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.perf_counter() - t0
    stop.set()
    if watch: w.join(timeout=1)
    if errs: raise RuntimeError(f"{len(errs)} failed: {errs[:2]}")
    tot = sum(r["n"] for r in recs)
    # decode window per stream = [end - decode_ms, end]; aggregate over the union
    starts = [r["end"] - r["dec_ms"]/1000.0 for r in recs]
    span = max(r["end"] for r in recs) - min(starts)
    return {"decode_sum": sum(r["tps"] for r in recs),      # comparable to repo S_TG
            "overlap": tot/span if span > 0 else 0.0,        # union of decode windows
            "wall": tot/wall,                                # what an app experiences
            "per_stream": st.mean(r["tps"] for r in recs),
            "peak": peak[0]}

pool = LocalPool(BASE, api_key=os.environ.get("LLAMA_API_KEY"))
print(f"attached: slots={pool.slots}  ctx_per_slot={pool.ctx_per_slot}  n_predict={N_PRED}")
procs_before = llama_procs()

print("\n[rule 5] first run discarded:", f"{run_batch(pool,1)['decode_sum']:.2f} t/s decode")

print("\n--- baseline: single stream ---")
s = [run_batch(pool, 1) for _ in range(3)]
for i, r in enumerate(s): print(f"  run {i+1}: decode {r['decode_sum']:.2f}  wall-metric {r['wall']:.2f} t/s")
base_dec = st.median(r["decode_sum"] for r in s)
base_wall = st.median(r["wall"] for r in s)
print(f"  median decode {base_dec:.2f} t/s   median wall-metric {base_wall:.2f} t/s")

print("\n--- 2 concurrent (= --parallel 2) ---")
two, ctrl = [], []
for i in range(3):
    r = run_batch(pool, 2, watch=True); two.append(r)
    print(f"  run {i+1}: decode_sum {r['decode_sum']:.2f}  overlap {r['overlap']:.2f}  "
          f"wall {r['wall']:.2f}  per-stream {r['per_stream']:.2f}  peak slots {r['peak']}")
    c = run_batch(pool, 1); ctrl.append(c["decode_sum"])       # rule 3 control
    print(f"    control single decode: {c['decode_sum']:.2f}")
d2 = st.median(r["decode_sum"] for r in two)
d2_wall = st.median(r["wall"] for r in two)
o2 = st.median(r["overlap"] for r in two)
peak2 = max(r["peak"] for r in two)

print("\n--- 4 concurrent against 2 slots (2 must queue) ---")
r4 = run_batch(pool, 4, watch=True)
print(f"  decode_sum {r4['decode_sum']:.2f}  overlap {r4['overlap']:.2f}  "
      f"wall {r4['wall']:.2f}  peak slots {r4['peak']}")
procs_after = llama_procs()

print(f"\ncontrol drift across session: {min(ctrl):.2f} - {max(ctrl):.2f} t/s "
      f"({(max(ctrl)-min(ctrl))/st.median(ctrl)*100:.1f}% spread)")

print("\n--- degradation detection (rule 8) ---")
import local_pool as LP
orig, saved = LP.DEGRADED_TPS, pool.degraded_reason
LP.DEGRADED_TPS = 10_000.0
try:
    pool.complete(prompt(0), n_predict=32); deg_ok = False; print("  FAIL served sub-floor result")
except Degraded as e:
    deg_ok = True; print(f"  PASS refused: {str(e)[:66]}...")
finally:
    LP.DEGRADED_TPS, pool.degraded_reason = orig, saved

ratio = d2 / base_dec
print("\n================ S3 =================")
ok = True
def crit(n_, c, d):
    global ok; print(f"  {'PASS' if c else 'FAIL'}  {n_}: {d}"); ok &= bool(c)
crit("P-S3a peak busy slots <= 2 at 4 concurrent", r4["peak"] <= 2, f"peak={r4['peak']} of {pool.slots}")
crit("P-S3b decode aggregate >= 1.3x single", ratio >= 1.3, f"{d2:.2f}/{base_dec:.2f} = {ratio:.2f}x (repo ceiling 1.48x)")
# P-S3c RETRACTED, not softened. It assumed client concurrency == server slot
# count. The repo's "B=4 is worse than B=2" is a --parallel 4 result; reproducing
# it needs a server restart, which the residency constraint forbids. Also
# decode_sum is only meaningful for n <= slots: at n=4 it sums four streams that
# never ran simultaneously. The --parallel 4 comparison stays untested here and
# rests on the repo's existing measurement (B=2 58.15 vs B=4 55.25 total t/s).
# What IS testable, and is what the scheduler must guarantee:
crit("P-S3c' client concurrency above slot count queues without collapse",
     r4["peak"] <= pool.slots and r4["wall"] >= d2_wall * 0.95,
     f"peak {r4['peak']}<={pool.slots}, wall {r4['wall']:.2f} vs 2c {d2_wall:.2f}")
crit("no extra llama processes", procs_after <= procs_before, f"{procs_before} -> {procs_after}")
crit("degradation refused, not served", deg_ok, "Degraded raised")
print("=====================================")
sys.exit(0 if ok else 1)
