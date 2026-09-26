# Qwen3.8-27B on gfx906

`general.architecture = qwen35`。65 層中 49 層が線形アテンション、16 層のみ
full attention という混合構成。MTP ヘッドつき。

| | 当初 | 最終 | |
|---|---|---|---|
| decode | 13.27 t/s | **26.26 t/s** | +98% |
| prefill | 98.96 t/s | **173.19 t/s** | +75% |

当初は Windows + Ollama + Vulkan。最終は Ubuntu + llama.cpp (パッチ 7 行)。

## 使い方

```bash
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 \
  llama-server -m models/Qwen3.8-27B-UD-Q4_K_S.gguf \
    -ngl 99 -c 32768 --parallel 2 -ctk q8_0 -ctv q8_0 --jinja
```

**`GGML_VK_DISABLE_*` はこれ以外を設定しないこと。** `--jinja` は tool calling に必要。

`-c 24576` を f16 KV で通すと VRAM 余裕が 28 MB しか残らない。**KV を q8_0 に
すればコンテキストを 33% 増やしてなお余裕が 17 倍**になり、速度は同じ
(詳細は [docs/serving.md](docs/serving.md))。長文プロンプト中心なら
`Q3_K_L` (14.12GB) で prefill 208.7。

OpenAI 互換 API として常駐させる手順は [docs/serving.md](docs/serving.md)。

## 量子化の選び方

**VRAM に収まる最大を選ぶ。** 小さくしても decode は速くならず、品質だけ落ちる。

| 形式 | GB | decode | PPL | HellaSwag |
|---|---|---|---|---|
| UD-Q2_K_XL | 9.83 | 24.73 | 6.2641 | 81.00% |
| UD-Q3_K_XL | 13.15 | 24.06 | 6.0266 | — |
| UD-IQ4_XS | 14.25 | 24.56 | 5.9474 | — |
| **UD-Q4_K_S** | 15.36 | **25.85** | **5.9350** | **83.25%** |

低ビットほど逆量子化コストが相対的に重く、到達帯域が下がってバイト削減と相殺する。

## 実用性

```
持続負荷    25分/255生成で -0.6%、junction 96C、定期再起動は不要
文脈長      実用上限 27k。5.5 倍に伸ばして decode 劣化 12%
並列処理    --parallel 2 で総スループット 1.48 倍 (4 では低下)
エージェント  47/48、1 ターン 7 秒、thinking トークン 0
```

**エージェント運用時の注意:** 時刻が未指定だと 10:00 を捏造する (量子化に依らず)。
他 11 種類の引数欠落では正しく聞き返すので、時刻の既定値を禁止する
システムプロンプトを入れること。

## ドキュメント

| | |
|---|---|
| [docs/setup.md](docs/setup.md) | ビルドとモデル取得 |
| [docs/findings.md](docs/findings.md) | 採用した 6 件と根拠 |
| [docs/rejected.md](docs/rejected.md) | 棄却した 17 件と理由 |
| [docs/benchmarks.md](docs/benchmarks.md) | 品質・下流タスク・エージェント評価 |
| [docs/predictions.md](docs/predictions.md) | 予測と結果の答え合わせ |
| [docs/serving.md](docs/serving.md) | OpenAI 互換 API として常駐させる |
| [docs/session-2026-09-25.md](docs/session-2026-09-25.md) | 進行中: decode のカーネル別内訳、GPU 未検証の試作4件 |

未解決は 1 件 — `q8_0 m=48 k=5120` の matvec が実効 16 GB/s (matvec 時間の 5.1%)。
m=48 では 60 CU を埋められず、split-K が要る。期待 +3〜4%。

施策4 (HOST_VISIBLE_VIDMEM) の効果量について、2026-09-25 のセッションで
本欄より大きい倍率 (2.3〜2.5倍) が測定された。対照なしの単発差のため未確定 —
詳細は [docs/session-2026-09-25.md](docs/session-2026-09-25.md)。
