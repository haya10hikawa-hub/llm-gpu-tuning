# Bonsai 2 27B on gfx906

Qwen3.8-27B を三値 (ternary) で量子化認識学習したモデル。構造は同じ
`qwen35` (64 層中 48 層が Gated DeltaNet) だが、重みの形式と演算経路が違う。

- 形式は `PQ2_0` (7.21 GB) と `PTQ1_0` (5.95 GB)。上流 llama.cpp では動かず、
  [PrismML のフォーク](https://github.com/PrismML-Eng/llama.cpp) が要る
- 重みはアダマール回転済みで、推論中に FWHT (高速アダマール変換) が走る

| | 当初 | 現在 | |
|---|---|---|---|
| decode | 8.74 t/s | **31.38 t/s** | 3.6 倍 |
| prefill | 59.58 t/s | **105.14 t/s** | +76% |

違いは環境変数 1 つだけ。出力は一致した。

## 使い方

```bash
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 \
  llama-server -m models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    -ngl 99 -c 65536 --parallel 1 -ctk q4_0 -ctv q4_0 --jinja
```

**この環境変数は必須。** Qwen3.8-27B では +20% だったが、こちらは 3.6 倍効く。
tool calling を使うなら [patches/toolcall-optional-newlines.patch](../patches/toolcall-optional-newlines.patch)
も当てること (理由は [docs/findings.md](docs/findings.md))。

## decode の内訳

`GGML_VK_PERF_LOGGER` による演算別の GPU 時間。1 トークン約 31 ms (深さ 4096) のうち:

```
三値 matvec   16.0 ms  51%   約 450 GB/s
FWHT           3.7 ms  12%   1 回 14 us。行が少なく GPU を埋められない
小さな演算     約 9 ms        起動 1 回 4〜12 us が約 1,500 回
Flash Attn     1.3 ms         深さ 32768 で 6.4 ms に伸びる
GDN            0.9 ms
```

## ドキュメント

| | |
|---|---|
| [docs/findings.md](docs/findings.md) | 採用した 2 件と根拠 |
| [docs/rejected.md](docs/rejected.md) | 効かなかったサーバー設定 |

エージェント評価 (MQL 課題・12 題) は実行中。途中経過は
[results/agent_mql.csv](results/agent_mql.csv)。
