# Radeon VII (gfx906) で LLM 推論を速くする

行列演算ユニットを持たない AMD Radeon VII (Vega 20 / gfx906) で Qwen3.8-27B を
動かすための計測・最適化一式。**コード変更 7 行で decode が 2 倍**になった。

| | 当初 | 最終 | |
|---|---|---|---|
| decode | 13.27 t/s | **26.26 t/s** | +98% |
| prefill | 98.96 t/s | **173.19 t/s** | +75% |

## 使い方

```bash
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 \
  llama-server -m Qwen3.8-27B-UD-Q4_K_S.gguf \
    -ngl 99 -c 24576 --parallel 2 --jinja
```

**`GGML_VK_DISABLE_*` はこれ以外を設定しないこと。** 特に
`DISABLE_INTEGER_DOT_PRODUCT` と `DISABLE_MMVQ` は大きく性能を落とす。

長文プロンプト中心なら `Q3_K_L` (14.12GB) で prefill 208.7 t/s。

## 効いたこと / 効かなかったこと

効いたのは 6 件だけ。Ollama をやめる (+74%)、環境変数の誤設定を外す
(prefill +30%)、VRAM に収まる最大の量子化を選ぶ、`DISABLE_HOST_VISIBLE_VIDMEM`
(+20%)、そして f16 累算を f32 に変える 7 行のパッチ (prefill +30%)。

棄却は 17 件。クロック固定、提出粒度、投機デコード、AMDVLK、カーネルの
occupancy 改善 — すべて実測で効果なしまたは逆効果だった。

→ [docs/findings.md](docs/findings.md) / [docs/rejected.md](docs/rejected.md)

## 注意

**`llama-bench` の token-generation 値と `test-backend-ops` の GFLOPS は
このハードウェアで信用できない。** 前者は実生成と 2.8 倍乖離し、後者は
23 型中 15 型が PCIe 転送律速で GPU を測っていない。

→ [docs/measurement.md](docs/measurement.md)

## 構成

| パス | 内容 |
|---|---|
| [docs/setup.md](docs/setup.md) | ビルドとモデル取得 |
| [docs/findings.md](docs/findings.md) | 採用した 6 件 |
| [docs/rejected.md](docs/rejected.md) | 棄却した 17 件 |
| [docs/architecture.md](docs/architecture.md) | ハードウェア特性と律速要因 |
| [docs/measurement.md](docs/measurement.md) | 測定の落とし穴と規約 |
| [docs/benchmarks.md](docs/benchmarks.md) | 品質・下流タスク・エージェント評価 |
| [AGENTS.md](AGENTS.md) | Agent 向け作業規約 (英語) |
| `*.py` `*.sh` | 計測スクリプト |
| `results/` `results27b/` | 全測定結果 (CSV) |

モデル (100GB) は追跡外。`fetch_models*.sh` で再取得する。

## パッチ

llama.cpp `3cf0325` に対する 1 件。GCN には行列コアが無く、コンパイラが
スカラ `float16_t` の累算器をペアに詰めないため、f16 累算は変換コストだけを
生む。`ggml_vk_get_mul_mat_mat_f16acc()` を GCN で false にする。

I-quant テンソルのみ -28.6%、K-quant は +0.9% (MMQ 経路は元から f16acc 非対象)。
