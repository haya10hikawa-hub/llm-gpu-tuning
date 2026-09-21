# ハードウェア特性と律速要因

## 実機確認した事実

```
GPU        Vega 20 [1002:66AF] Radeon VII, 16 GB HBM2 (4096-bit, 1024 GB/s)
llama.cpp  matrix cores: none | warp size: 64 | int dot: 1 | fp16: 1 | LDS 65536
coopmat    非対応 (llvmpipe のみ公開)
subgroup   min = max = 64 (wave64 固定、wave32 不可)
int dot    integerDotProduct4x8BitPackedSignedAccelerated = true
PCIe       Gen3 x16、実測 10.4 GB/s
ROCm       gfx906 は 5.7 でサポート打ち切り。Vulkan が唯一の経路
```

行列ユニットの代替は `v_dot4_i32_i8` (53.8 TOPS)。`v_dot2_f32_f16` (27.7 TFLOPS)
ではない — `DISABLE_DOT2` が性能に影響しないことで確認済み。

## モデル構造 (Qwen3.8-27B)

```
block_count             65   (線形アテンション 49層 + full attention 16層)
full_attention_interval  4
embedding_length      5120 / feed_forward 17408
attention.head_count    24 / head_count_kv 4 / key = value_length 256
ssm.state_size         128 / inner_size 6144 / time_step_rank 48
nextn_predict_layers     1   (MTP ヘッドあり)
再帰状態: ssm 3.15 MB + conv 0.12 MB / 層 → 49層で 160 MB
```

## 律速要因

外部観測 (クロック感度) と内部観測 (パイプライン統計) が一致した。

### クロック感度

sclk と mclk を独立に振り、decode の弾性を測定した。

```
mclk 1000 → 800  (-20%)   26.17 → 25.01  (-4.4%)   弾性 0.22
sclk 1801 → 1683 (-6.6%)  26.23 → 25.50  (-2.8%)   弾性 0.42
```

**動作点近傍で sclk が mclk の約 1.9 倍効く。演算律速寄り。**

大きな摂動では両方とも弾性 0.7〜0.85 に近づく。どちらの資源も落としきれば
律速側に回るため、判定に使えるのは小摂動の弾性のみ。

### パイプライン統計

`GGML_VK_PIPELINE_STATS=mul_mat_vec` + `test-backend-ops test -o MUL_MAT`

| type | VGPR | waves/SIMD | code | LDS | 実測 GB/s |
|---|---|---|---|---|---|
| q6_k | 256 | 1 | 41864 | 512 | 343 |
| iq4_xs | 256 | 1 | 88672 | 512 | 489 |
| iq3_s | 256 | 1 | 51396 | 2048 | 287 |
| q3_k | 128 | 2 | 17132 | 512 | 353 |
| **q4_k** | **64** | **4** | 6732 | 0 | **618** |
| **q5_k** | **64** | **4** | 7760 | 0 | **513** |

GCN の SIMD はレーンあたり 256 VGPR。256 使えば 1 wave しか走らず、
HBM のレイテンシ (400〜800 cycle) を隠せない。

**これは wave64 固有の問題。**

```
gfx906 (GCN,  wave64) : SIMD あたり 256 VGPR → 256 使用で 1 wave
RDNA   (      wave32) : SIMD あたり 512 VGPR → 同じシェーダで 4 wave
```

シェーダが RDNA のレジスタファイルを前提にチューニングされているため、
wave64 では occupancy が 1/4 に落ちる。「最新の専用最適化パスほど gfx906 で
遅い」という現象の機序。

**ただし occupancy を改善しても速くならなかった** (→ `rejected.md`)。
相関はあるが因果ではない。

## 量子化型と性能の 2 軸

帯域を決めるのは K/I の別ではなく bits-per-weight。

```
q3_K   3.44 bpw → 353 GB/s        iq3_s  3.44 bpw → 287 GB/s
iq4_xs 4.25 bpw → 489 GB/s        q4_K   4.50 bpw → 618 GB/s
```

| 軸 | 影響先 | 内容 |
|---|---|---|
| bits-per-weight | decode | 低ビットほど逆量子化が相対的に重く到達帯域が下がる。バイト削減と相殺 |
| MMQ 経路の可否 | prefill | I-quant は `q8_1` + `v_dot4_i32_i8` の GEMM に乗れない |

## 文脈長の上限

full-attention 層の KV は 64 KB/token (65層中16層のみ)。

| モデル | 余裕 | 最大 ctx |
|---|---|---|
| UD-Q4_K_S 15.36GB | 1.80 GB | 約 27k |
| UD-Q3_K_XL 13.15GB | 4.01 GB | 約 61k |

モデルの `context_length = 262144` は 16GB では届かない。ただし通常の
Transformer なら全65層で伸びて 260 KB/token となり上限は 1/4。線形アテンションの
おかげで**同じ VRAM に 4 倍長い文脈が収まる**。
