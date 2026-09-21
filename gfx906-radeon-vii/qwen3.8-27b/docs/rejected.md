# 棄却した施策

再試行を防ぐための記録。すべて実測で効果なしまたは逆効果と確認済み。

| 施策 | 結果 | 理由 |
|---|---|---|
| GPU クロック固定 | 効果なし | `auto` が負荷時に既に 1000MHz へ昇圧 |
| `MAX_NODES_PER_SUBMIT` 11水準 | ±3% | 既定 100 が最適。`=1` で -14%、`SERIALIZE` で -34% |
| flash attn × ubatch × KV量子化 12構成 | ±0.6% | 全構成で差なし |
| MTP 投機デコード | 最良 +6.6% | n_max=2 で 9.07、4 で 7.06、8 で 3.31。受理率が急落 |
| ngram 投機デコード | ドラフト 0 件 | 機能せず |
| AMDVLK ドライバ | 起動不可 | kernel 7.0 で `ERROR_INCOMPATIBLE_DRIVER` |
| `get_rows` ベクトル化 | 0% | カーネル時間 150.6 → 156.7 us。正当性 240/240 合格だが無効 |
| 再帰状態の in-place 化 | 着手せず | クリーン状態では GET_ROWS+CPY で 3.3% のみ |
| IQ4_XS 向け q8_1 MMVQ 追加 | 着手せず | decode の I/K 差は 455 vs 495 GB/s = 8% のみ |
| `FORCE_MMVQ=1` | -1.9% | AMD は `k >= 2048` で既定 true。no-op |
| `DISABLE_DOT2=1` | -0.2% | `v_dot2_f32_f16` 経路は使われていない |
| `ENABLE_MEMORY_PRIORITY` / `ASYNC_USE_TRANSFER_QUEUE` | ±1% | 中立 |
| `ALLOW_GRAPHICS_QUEUE=1` | +1.6% | ノイズ範囲。8B では +3.8%、27B では -0.1% |
| `DISABLE_FUSION` / `GRAPH_OPTIMIZE` / `MULTI_ADD` | -0.3 ~ -4% | 既定 (有効) が最適 |
| matvec の occupancy 改善 (`rm_kq` 4→2) | 0% | q3_k を 2→3 wave にしても 26.19 → 26.19 |
| matvec のスピル削減 (`rm_iq` 8→2) | -20 ~ -59% | バッチ decode が壊滅的に悪化 |
| 純 K-quant モデル (decode 目的) | +4.1% | 期待 +22% に届かず。prefill には +22.2% 有効 |

## 最大の空振り — matvec の occupancy

「全形状が 650 GB/s に達すれば decode +26%」と見積もったが、3 つの独立した
実験で実現不能と確認した。

**occupancy と帯域の相関は因果ではなかった。** 両者が「逆量子化の複雑さ」という
第三の変数に従属していただけ。occupancy 向上分は活性化ベクトルの再読み込み
増加でちょうど相殺される。

```
rm_kq=4 (現行)  q3_k VGPR 128 → 2 wave/SIMD   decode 26.19
rm_kq=2         q3_k VGPR  84 → 3 wave/SIMD   decode 26.19
```

スピル側はさらに明確に逆効果:

```
             B=1     B=2     B=4
rm_iq=8 (現行) 24.67   36.53   32.54
rm_iq=2      19.67   22.50   13.49
```

llama.cpp の GCN 向けチューニング (`rm_kq=4, rm_iq=8`) は正しい。
matvec に設定レベルの余地はない。

## 純 K-quant (非 UD)

同サイズ対照 (UD-IQ4_XS 14.25GB vs bartowski Q3_K_L 14.12GB):

| | prefill | decode | PPL |
|---|---|---|---|
| UD-IQ4_XS | 170.8 | 24.89 | 5.9474 |
| Q3_K_L | **208.7 (+22.2%)** | 25.91 (+4.1%) | 6.0442 (+1.63%) |

decode が伸びなかった理由は帯域ではなく転送量。純 K-quant の加重平均帯域は
432.6 GB/s で混在版の 473.9 GB/s より**低い**。転送量が 30% 少ないぶんだけ速い。
