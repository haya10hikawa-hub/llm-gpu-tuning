# Instructions for agents working in this repo

Optimization and benchmarking of LLM inference on an AMD Radeon VII (Vega 20 /
gfx906) — a GPU with **no matrix cores**, wave64 only, no cooperative matrix.
Target model: Qwen3.8-27B (`general.architecture = qwen35`, hybrid attention).

Read `docs/` before proposing changes. Every claim there is backed by a
measurement stored under `results/` or `results27b/`.

## Measurement rules — violating these produces wrong data

1. **One GPU job at a time.** Check `gpu_busy_percent` and `mem_info_vram_used`
   before starting. Two processes each claiming 13+ GB exhausts VRAM, resets the
   GPU, and silently drops every later run to CPU.
2. **Check every run for `no usable GPU`.** CPU fallback only prints a warning
   and then reports plausible-looking numbers (~0.85 t/s for 27B).
3. **Interleave a control config every 3 runs.** Repeated process start/stop
   fragments the VRAM allocator; decode drifts 24 -> 14.5 t/s over hours.
4. **Write a falsifiable prediction before changing anything.** If it misses,
   stop and re-diagnose instead of continuing.
5. **Treat the first run separately** — it pays shader compilation cost.
6. **Keep effect-size predictions inside the measured range.** Say so explicitly
   when extrapolating. Two of four extrapolations here were wrong.
7. **When a number does not move, suspect saturation or clamping first**, not
   "no effect". VGPR counts clamp at 256 on GCN and the overflow shows up as
   scratch spill instead.
8. **State the success condition for every measurement.** Two failures here
   looked like partial success: `prefill` reported a healthy number while
   `decode` was 0 tokens, and `curl -s` treated HTTP 503 (model still loading)
   as success.
9. **Never conclude from a single differing case.** Widen the category and check
   that it reproduces. One n=1 "finding" here did not replicate.

## Tools that must not be used for performance claims

| Tool | Why |
|---|---|
| `llama-bench` token-generation numbers | 2.8x off from real generation on this hybrid-attention model (8.59 vs 24 t/s) |
| `test-backend-ops perf` GFLOPS | 15 of 23 types are PCIe-transfer bound, pinned at 10.4 GB/s regardless of size |

Use `llama-completion` or `llama-server` real generation, and
`GGML_VK_PERF_LOGGER=1` for per-op GPU time. `test-backend-ops test` is fine for
correctness only.

## Environment hazards

- `pgrep -f` / `pkill -f` match **your own command line**, including strings
  inside `echo`. Use a bracket pattern: `pgrep -f "bin/llama-comp[l]etion"`.
  This killed the running shell four times during this work.
- `llama-completion` enters an interactive turn after generating. In batch
  measurement always pass `-st` and `< /dev/null`, or it is killed by timeout
  before printing timings.
- After a GPU reset the DRM nodes are recreated and lose the user ACL. Recover
  with `sudo setfacl -m u:$USER:rw /dev/dri/renderD128 /dev/dri/card1`.
- `git add -A` will hash multi-GB GGUF files if they are not yet ignored. Check
  `.gitignore` first.

## Conventions

- Human-facing documentation (`README.md`, `docs/*.md`) is Japanese and fits in
  one scroll each.
- Anything an agent or a prompt reads repeatedly — this file, script comments,
  eval task definitions — is English.
- Scripts write results as CSV under `results/` or `results27b/`. Keep that.
- Model files are not tracked. `fetch_models*.sh` re-downloads them.
