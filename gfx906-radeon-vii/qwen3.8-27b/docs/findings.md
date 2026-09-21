# 採用した最適化

すべて実測値つき。詳細な根拠は `results/` の CSV にある。

| # | 施策 | 効果 | 根拠 |
|---|---|---|---|
| 1 | Ollama → llama.cpp 直接 | +74% | 同一モデル・同一GPU で Ollama 8.4 / llama.cpp 14.6 |
| 2 | `DISABLE_INTEGER_DOT_PRODUCT` を**設定しない** | prefill +30% | 27B: 132.9 → 102.1 / 8B: 742.8 → 360.9 |
| 3 | `DISABLE_MMVQ` を**設定しない** | decode +13% | 26.28 → 23.27 |
| 4 | `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | +20% | Q4_K_S: 21.77 → 26.11 |
| 5 | VRAM に収まる最大の量子化 | +7% | Q2_K_XL 24.73 / Q4_K_S 25.85 |
| 6 | f16acc パッチ (7行) | prefill +30% | 132.90 → 173.19、decode 不変 |

## 2 と 3 — 環境変数の誤設定

RADV は gfx906 で `integerDotProduct4x8BitPackedSignedAccelerated = true` を
公開している。これを切ると K-quant が MMQ 経路 (`q8_1` + `v_dot4_i32_i8`) を
失い、全型が汎用 FP16 経路に落ちる。

## 4 — host-visible VRAM

Radeon VII は Resizable BAR 非対応で host-visible VRAM が 256MB しかない。
VRAM 上限に近いモデルほどアロケータがここにバッファを置く確率が上がる。

| モデル | VRAM 余裕 | 効果 |
|---|---|---|
| UD-Q3_K_XL 13.15GB | 3.2 GB | +3.6% |
| UD-Q4_K_S 15.36GB | 1.0 GB | **+20%** |

## 5 — 量子化サイズと速度・品質

**小さくしても速くならない。** 低ビットほど逆量子化コストが相対的に重く、
到達帯域が下がってバイト削減と相殺する。

| 形式 | GB | decode | PPL | 劣化 |
|---|---|---|---|---|
| UD-Q2_K_XL | 9.83 | 24.73 | 6.2641 | +5.55% |
| UD-IQ3_XXS | 10.93 | 23.99 | 6.1265 | +3.23% |
| UD-Q3_K_XL | 13.15 | 24.06 | 6.0266 | +1.54% |
| UD-IQ4_XS | 14.25 | 24.56 | 5.9474 | +0.21% |
| **UD-Q4_K_S** | 15.36 | **25.85** | **5.9350** | 0.00% |

速度・品質とも最大が最良。小さい量子化を選ぶ理由は「VRAM に載らない」以外にない。

## 6 — f16acc パッチ

`ggml_vk_get_mul_mat_mat_f16acc()` は coopmat 非対応デバイスで f16 累算を選ぶ。
GCN には行列コアが無く、ACO がスカラ `float16_t` の累算器をペアに詰めないため、
レート利得ゼロで変換コストだけが乗る。

```cpp
if (ctx->device->architecture == vk_device_architecture::AMD_GCN) {
    return false;
}
```

型別 MUL_MAT の実測 (prefill):

```
iq4_xs  2975.5 → 2071.2 ms  -30.4%   I-quant (MMQ 不可 → f16acc 有効だった)
iq3_xxs  156.1 →  106.9 ms  -31.5%
q5_K     824.0 →  833.4 ms   +1.1%   K-quant (MMQ 経路 → f16acc 元から無効)
q4_K     779.6 →  784.6 ms   +0.6%
```

`AMD_GCN` は `minSubgroupSize == maxSubgroupSize == 64` で判定。RDNA 以降は非該当。
Vega 10 / Polaris / Fiji でも同じ利得が期待できる (未検証)。

## バッチ処理

| 並列数 | S_TG t/s | 総合 t/s |
|---|---|---|
| 1 | 24.67 | 37.19 |
| **2** | **36.53** | **58.15** |
| 4 | 32.54 | 55.25 |

B=2 で総スループット 1.48 倍。B=4 で低下するので `--parallel 2` が最適。
