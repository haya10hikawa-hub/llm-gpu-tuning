# Instructions for agents working in this repo

A hub of per-GPU, per-model LLM inference optimization records. Every claim in
`<gpu>/<model>/docs/` is backed by a CSV under the matching `results/`.

Before working on a specific GPU, also read `<gpu>/AGENTS.md` — it lists the
hazards specific to that hardware.

## Measurement rules

These exist because each one was violated during the first campaign in this
repo, and each violation produced a wrong conclusion.

1. **One GPU job at a time.** Check GPU busy and VRAM before starting. Two
   processes each claiming most of VRAM exhausts it, resets the GPU, and
   silently drops every later run to CPU.
2. **Check every run for a CPU-fallback warning.** Fallback only prints a
   warning and then reports plausible-looking numbers.
3. **Interleave a control config every 3 runs.** Long measurement sessions
   drift; without a control you cannot tell drift from effect.
4. **Write a falsifiable prediction before changing anything.** If it misses,
   stop and re-diagnose instead of continuing.
5. **Treat the first run separately** — it pays shader/kernel compilation cost.
6. **Keep effect-size predictions inside the measured range.** Say so
   explicitly when extrapolating. Mechanism predictions tend to hold; effect
   sizes extrapolated from partial measurements tend not to.
7. **When a number does not move, suspect saturation or clamping first**, not
   "no effect". A clamped counter hides the real signal somewhere else.
8. **State the success condition for every measurement.** Partial success looks
   like success: a healthy prefill number next to zero generated tokens, or an
   HTTP client treating 503 as OK.
9. **Never conclude from a single differing case.** Widen the category and
   check that it reproduces.

## Reporting

- Record rejected approaches with the number that rejected them. A future agent
  must be able to see that an idea was already tried.
- Record prediction vs outcome. The ratio of hits to misses is itself data.
- Do not present a measurement taken on a degraded system as a property of the
  workload.

## Conventions

- Human-facing documentation (`README.md`, `docs/*.md`) is Japanese and fits in
  one scroll each.
- Anything an agent or a prompt reads repeatedly — this file, per-GPU
  `AGENTS.md`, script comments, eval task definitions, output labels — is
  English.
- Harnesses in `tools/` take paths from the environment:
  `LLM_TUNING_ROOT`, `LLAMA_BIN`, `BENCH_MODELS`, `BENCH_RESULTS`.
- Model weights are never tracked. Each model directory has a fetch script.
