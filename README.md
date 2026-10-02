# llm-gpu-tuning

GPU × LLM モデルの組み合わせごとに、**何が効いて何が効かなかったか**を実測値
つきで蓄積するハブ。推測ではなく測定で判断し、棄却した施策も理由つきで残す。

## 収録

| GPU | Model | Prefill & Decode (Only / MTP) | Patch |
|---|---|---|---|
| [gfx906-radeon-vii](gfx906-radeon-vii/) | [Qwen3.8-27B](gfx906-radeon-vii/qwen3.8-27b/) | **Only:** prefill 98.96 → **173.19 t/s** / decode 13.27 → **26.26 t/s**<br>**MTP:** — | [`gfx906-radeon-vii/patches/gcn-f16acc.patch`](gfx906-radeon-vii/patches/gcn-f16acc.patch) |
| | [Qwen3-8B](gfx906-radeon-vii/qwen3-8b/) | **Only:** prefill 360.93 → **742.80 t/s** / decode 69.52 → **78.63 t/s**（誤設定 → default）<br>**MTP:** — | [`gfx906-radeon-vii/patches/gcn-f16acc.patch`](gfx906-radeon-vii/patches/gcn-f16acc.patch) |
| | [K2-Horizon-7B](gfx906-radeon-vii/k2-horizon-7b/) | **Only:** prefill **628.5 t/s** / decode **76.6 t/s**（Q4_K_M、単発）<br>**MTP:** — | — |
| | [MiMo-V2.6-Distill-Qwen-9B](gfx906-radeon-vii/mimo-v2.6-distill-qwen-9b/) | **Only:** prefill **743.66 t/s**（Q8_0、単発最大）/ decode **74.61 t/s**（Q5_K_M、単発最大）<br>**MTP:** — | — |
| | [Bonsai 2 27B](gfx906-radeon-vii/bonsai-2-27b/) | **Only:** prefill 59.6 → **284 t/s** / decode 8.74 → **約 59 t/s**<br>**MTP:** decode **約 72 t/s**（draft 2） | [`gfx906-radeon-vii/bonsai-2-27b/patches/`](gfx906-radeon-vii/bonsai-2-27b/patches/) / [`toolcall-optional-newlines.patch`](gfx906-radeon-vii/patches/toolcall-optional-newlines.patch) |


## 構成

```
AGENTS.md                  全 GPU 共通の測定規約 (英語)
tools/                     GPU・モデル非依存のハーネス
<gpu>/
  README.md                ハードウェア特性の要約
  AGENTS.md                その GPU 固有の罠 (英語)
  docs/                    アーキテクチャ・測定手法
  patches/                 上流に出せるパッチ
  <model>/
    README.md              その組み合わせの結果
    docs/                  採用・棄却・ベンチマーク・予測記録
    results/               全測定結果 (CSV)
```

## 方針

**測定を信用する前に測定系を疑う。** 本リポジトリの最初の作業では、VRAM 断片化
という測定環境の異常を対象の構造的欠陥と誤認し、無駄なカーネル改修を 1 件
実施した。「状態の読み書きが decode の 36.6%」という診断は、クリーンな状態では
3.3% だった。

そのため各ディレクトリに以下を残す。

- **棄却記録** — 再試行を防ぐため、効かなかった施策と理由を残す
- **予測記録** — 変更前に反証可能な予測を書き、当たり外れを記録する
- **測定規約** — 対照の挟み込み、飽和の確認、n=1 を結論にしないこと

## 追加するとき

新しい GPU なら `<gpu>/` を、既存 GPU の新しいモデルなら `<gpu>/<model>/` を
作る。`tools/` のハーネスは環境変数でパスを受けるので再利用できる。

```bash
LLM_TUNING_ROOT=<gpu>/<model> LLAMA_BIN=<path> python3 tools/agentic_eval.py --server ...
```

作業前に [AGENTS.md](AGENTS.md) と対象 GPU の `AGENTS.md` を読むこと。
