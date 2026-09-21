# Qwen3-8B on gfx906 (比較用)

通常の Transformer (`general.architecture = qwen3`, 36 層)。
27B の混合アテンションとの対照として測定した。

## ここで最初に見つかったこと

**環境変数の誤設定が prefill を半減させる。**

| 構成 | prefill | decode |
|---|---|---|
| default | 742.80 | 78.63 |
| `DISABLE_INTEGER_DOT_PRODUCT=1` | **360.93 (-51%)** | 69.52 |
| `DISABLE_MMVQ=1` | 741.07 | 69.21 (-12%) |

27B では −23% だが、8B では −51%。decode の比率が低いぶん prefill の損失が
そのまま出る。

**`GGML_VK_DISABLE_F16=1` で prefill が上がる兆候もここで出ていた**
(742.80 → 776.26、+4.5%)。当時は追わなかったが、27B で +12.7% として再出現し、
`gcn-f16acc.patch` につながった。**異常値を放置しない判断が成果に直結した例。**

## 27B との違い

| | 8B (qwen3) | 27B (qwen35) |
|---|---|---|
| アテンション | 全層 full | 49/65 が線形 |
| `ALLOW_GRAPHICS_QUEUE` | +3.8% | −0.1% |
| KV / token | 全層で伸びる | 16 層のみ 64 KB/token |

`ALLOW_GRAPHICS_QUEUE` の符号が逆転する。**この手のノブはモデル依存**で、
GPU だけで決まらない。

## 測定結果

`results/` に Stage A (クロック 4 水準)、Stage B (env 10 水準)、
Stage C (量子化 5 × env 4)、Stage D (op 別 GFLOPS) を収録。

Stage D の GFLOPS 値は **PCIe 転送律速なので使えない** (詳細は
`../AGENTS.md`)。参考として残してある。

```bash
./fetch_models.sh    # Qwen3-8B 5形式 (25GB)
```
