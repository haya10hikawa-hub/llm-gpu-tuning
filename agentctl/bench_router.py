"""S2: routing decision latency. Criterion: p99 < 1 ms over 10k decisions.

Measurement rules from the repo AGENTS.md that apply here:
  rule 5 - the first run pays warmup, so it is reported separately
  rule 3 - a control config is interleaved to detect drift
"""
import random, statistics as st, sys, time
from router import Backend, Class, NoLegalBackend, Privacy, route

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

CLASSES = list(Class)
CONTROL = (Class.TOOL_CALL, Privacy.NORMAL, 500, True, None)


def make_case(rng):
    return (rng.choice(CLASSES),
            Privacy.STRICT if rng.random() < 0.2 else Privacy.NORMAL,
            rng.choice([100, 500, 4000, 11000, 12000, 40000]),
            rng.random() < 0.5,
            Backend.OPUS if rng.random() < 0.05 else None)


def timed(case):
    t0 = time.perf_counter_ns()
    try:
        route(*case)
    except NoLegalBackend:
        pass
    return time.perf_counter_ns() - t0


rng = random.Random(1)
cases = [make_case(rng) for _ in range(N)]

first = timed(cases[0])                      # rule 5: first run separately

samples, control = [], []
for i, c in enumerate(cases):
    samples.append(timed(c))
    if i % 3 == 2:                           # rule 3: control every 3 runs
        control.append(timed(CONTROL))

samples.sort(); control.sort()
p = lambda a, q: a[min(int(len(a) * q), len(a) - 1)]
us = lambda ns: ns / 1000.0

print(f"n={N}  (first run, reported separately: {us(first):.2f} us)")
print(f"  p50 {us(p(samples,.50)):7.2f} us   p90 {us(p(samples,.90)):7.2f} us"
      f"   p99 {us(p(samples,.99)):7.2f} us   max {us(samples[-1]):7.2f} us")
print(f"  mean {us(st.mean(samples)):.2f} us")
print(f"control n={len(control)}: p50 {us(p(control,.50)):.2f} us  "
      f"p99 {us(p(control,.99)):.2f} us   (drift check)")

p99_us = us(p(samples, .99))
budget_us = 1000.0
print()
print(f"CRITERION p99 < 1 ms : p99 = {p99_us:.2f} us -> {'PASS' if p99_us < budget_us else 'FAIL'}")
print(f"PREDICTION p99 < 200 us : {'HIT' if p99_us < 200 else 'MISS'} (was marked an extrapolation)")
sys.exit(0 if p99_us < budget_us else 1)
