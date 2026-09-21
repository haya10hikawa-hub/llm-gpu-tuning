# gfx906 (Radeon VII) — hardware-specific hazards

Read the repo-root `AGENTS.md` first. This file only covers what is specific to
this GPU.

## Tools that must not be used for performance claims here

| Tool | Why |
|---|---|
| `llama-bench` token-generation numbers | 2.8x off from real generation on hybrid-attention models (8.59 vs 24 t/s measured) |
| `test-backend-ops perf` GFLOPS | 15 of 23 quant types are PCIe-transfer bound, pinned at 10.4 GB/s across a 12x size range — it measures the PCIe link, not the GPU |

Use `llama-completion` or `llama-server` real generation.
`GGML_VK_PERF_LOGGER=1` gives per-op GPU time. `test-backend-ops test` is fine
for correctness only.

## Environment hazards

- **VRAM fragmentation.** Repeated process start/stop degrades decode by up to
  40% over hours (24 -> 14.5 t/s). A resident `llama-server` does **not**
  degrade — 25 minutes of continuous load moved it 0.6%. Interleave a control
  config to detect drift in measurement scripts.
- **GPU reset drops the DRM ACL.** After an out-of-memory reset the device nodes
  are recreated and the user loses access, so llama.cpp silently falls back to
  CPU (~0.85 t/s for a 27B). Recover with:
  `sudo setfacl -m u:$USER:rw /dev/dri/renderD128 /dev/dri/card1`
- **`llama-completion` blocks on stdin** after generating. In batch measurement
  always pass `-st` and `< /dev/null`, or timeout kills it before it prints
  timings.
- **`pgrep -f` / `pkill -f` match your own command line**, including strings
  inside `echo`. Use a bracket pattern: `pgrep -f "bin/llama-comp[l]etion"`.

## Clock control

`gpuclk.sh` pins clocks via `power_dpm_force_performance_level`. `auto` already
ramps to 1000 MHz mclk / 1801 MHz sclk under load, so pinning is for measurement
determinism, not speed. Always return to `auto` when done.

Decode elasticity at the operating point: **sclk 0.42, mclk 0.22** — compute
side matters roughly twice as much as bandwidth. Thermal headroom is ample
(junction peaks at 96C against a 110C limit), so throttling is not a constraint.
