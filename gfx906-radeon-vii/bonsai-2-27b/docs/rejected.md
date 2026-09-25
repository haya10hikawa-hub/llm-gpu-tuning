# 効かなかった施策

サーバーが llama-bench より遅い原因を探して試したもの。どれも差は 1% 以内で、
原因は host-visible VRAM だった (→ findings.md #1)。いずれも環境変数を
付ける前の測定。

| 施策 | decode |
|---|---|
| 既定 | 8.72 t/s |
| `-fa on` | 8.73 |
| `-ctxcp 0` (チェックポイント無効) | 8.67 |
| `-cram 0` (キャッシュ RAM 無効) | 8.73 |
| `-ub 256 -b 512` | 8.70 |
| `--fit off` | 8.77 |
| `--no-op-offload` | 8.76 |
| `-t 4 --poll 0` | 8.76 |
| `--no-host` | 8.78 |

推奨サンプリング (temp 0.7 / top_p 0.8 / top_k 20 / presence 1.5) は −7%
(8.14 t/s)。CPU でも GPU (`-bs`) でも同じだった。

生の値は `results/server_options.csv`。
