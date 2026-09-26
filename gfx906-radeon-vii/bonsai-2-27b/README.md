# Bonsai 2 27B on gfx906

Qwen3.8-27B を三値 (ternary) で量子化認識学習したモデル。構造は同じ
`qwen35` (64 層中 48 層が Gated DeltaNet) だが、重みの形式と演算経路が違う。

- 形式は `PQ2_0` (7.21 GB) と `PTQ1_0` (5.95 GB)。上流 llama.cpp では動かず、
  [PrismML のフォーク](https://github.com/PrismML-Eng/llama.cpp) が要る
- 重みはアダマール回転済みで、推論中に FWHT (高速アダマール変換) が走る

## 速度

評価と同じ設定 (ctx 65536, ub 256, b 512, 思考オフ, 貪欲)。単位は t/s。

| | decode 4.3k | decode 15k | prefill 4.3k | prefill 15k |
|---|---|---|---|---|
| 当初 | 8.74 | | 59.6 | |
| 環境変数のみ | 27.6 | 28.9 | 99.8 | 91.2 |
| **パッチ + KV f16** | **47.2** | **48.6** | **250.1** | **221.1** |

当初比で decode 5.4 倍、prefill 4.2 倍。パッチ前後で貪欲 128 トークンは一致し、
KL ダイバージェンスは decode 0.0010 / prefill 0.0038
([results/opt_e2e.csv](results/opt_e2e.csv), [results/kld.csv](results/kld.csv))。

エージェント用サンプリング (presence penalty 付き) では CPU 側の遅さも直し、
34.7 → 48.4 t/s ([results/agent_sampling.csv](results/agent_sampling.csv))。

## 使い方

```bash
cd <PrismML llama.cpp @842b188>
git am <このディレクトリ>/patches/*.patch   # 任意。速度を上げるパッチ一式
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 GGML_VK_RM_KQ_INT=2 \
  llama-server -m models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    -ngl 99 -c 65536 --parallel 1 -ctk f16 -ctv f16 --jinja
```

- **`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` は必須** (無いと 3.6 倍遅い。[docs/findings.md](docs/findings.md))
- tool calling を使うなら [patches/toolcall-optional-newlines.patch](../patches/toolcall-optional-newlines.patch) も当てる
- 思考を切るなら `--chat-template-kwargs '{"enable_thinking":false}'`。`--reasoning-budget 0` だけでは止まらない

## 理論上限までの距離 (decode)

1 トークンで読む量 (重み 6.85 GB ほか) を実測の読み出し上限 830 GB/s で割ると約 9 ms。

```
                現在      下限
重みの mat-vec   12.4 ms   8.3 ms   メモリ律速
それ以外          5.6 ms   0.7 ms   依存の段数 (同期 756 回/トークン)
CPU 側など        3.0 ms     -
合計             21.0 ms   9.0 ms   到達率 43%
```

## エージェント評価 (思考オン, 12 題)

11 題が全テスト合格、T07 のみ不合格 ([results/agent_tasks.csv](results/agent_tasks.csv))。
思考オフの再計測は実行中。

## ドキュメント

| | |
|---|---|
| [docs/optimizations.md](docs/optimizations.md) | パッチの効果と学んだこと |
| [docs/findings.md](docs/findings.md) | 環境変数と tool call 文法 |
| [docs/rejected.md](docs/rejected.md) | 効かなかった施策 |
