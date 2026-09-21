# gfx906 / AMD Radeon VII (Vega 20)

**行列演算ユニットを持たない GPU。** wave64 固定、cooperative matrix 非対応、
ROCm はサポート打ち切り済みで Vulkan が唯一の経路。

```
60 CU / 16 GB HBM2 (4096-bit, 1024 GB/s) / PCIe Gen3 x16 (実測 10.4 GB/s)
matrix cores: none | warp size: 64 | int dot: 1 | fp16: 1 | LDS 65536
coopmat 非対応 / subgroup min = max = 64 / ROCm 5.7 で打ち切り
```

行列ユニットの代替は **`v_dot4_i32_i8` (53.8 TOPS)**。`v_dot2_f32_f16`
(27.7 TFLOPS) ではない — `GGML_VK_DISABLE_DOT2` が性能に影響しないことで確認済み。

## この GPU 全体に効くこと

| 施策 | 効果 |
|---|---|
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` を**設定しない** | prefill が最大 2 倍変わる |
| `GGML_VK_DISABLE_MMVQ` を**設定しない** | decode −13% を回避 |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | VRAM 上限に近いモデルで +20% |
| [patches/gcn-f16acc.patch](patches/gcn-f16acc.patch) | prefill +30%。GCN 世代全般に効くはず |

## 固有の落とし穴

**Resizable BAR 非対応で host-visible VRAM が 256MB しかない。** VRAM 上限に
近いモデルほどアロケータがここにバッファを置き、大きく遅くなる。

**シェーダが RDNA のレジスタファイル (512 VGPR/SIMD, wave32) を前提に
チューニングされている。** wave64 の gfx906 は 256 VGPR/SIMD なので、同じ
シェーダで occupancy が 1/4 に落ちる。「最新の専用最適化パスほど遅い」現象の機序。
ただし occupancy を改善しても速くならなかった (相関はあるが因果ではない)。

詳細は [docs/architecture.md](docs/architecture.md)。
測定の罠は [docs/measurement.md](docs/measurement.md) と [AGENTS.md](AGENTS.md)。

## 収録モデル

| モデル | decode | prefill | 備考 |
|---|---|---|---|
| [Qwen3.8-27B](qwen3.8-27b/) | 26.26 t/s | 173.19 t/s | 混合線形アテンション (65層中49層) |
| [Qwen3-8B](qwen3-8b/) | — | — | 比較用。通常の Transformer |

## ツール

```bash
./gpuclk.sh high     # クロック固定 (計測時のみ)
./gpuclk.sh auto     # 復帰
./gpuclk.sh show     # 現在のクロック・温度・VRAM
```
