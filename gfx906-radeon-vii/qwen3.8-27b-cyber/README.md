# Qwen3.8-27B-Uncensored-Cyber on gfx906

`cyjin-yl/Qwen3.8-27B-Uncensored-Cyber-agentic-imatrix-GGUF` の IQ4_XS (MTP ヘッド付き 1 ファイル、15.38 GiB)。
取得は [fetch_models.sh](fetch_models.sh)。MTP なしで測るときも同じファイルを `--spec-type` なしで使う。

本体は IQ4_XS 10.76 GiB + Q5_K 1.67 GiB、出力層 Q8_0。1 トークンで 14.70 GB 読むので decode の上限は 56 t/s。

## 速度

KV q8_0、ub 1024、ctx 16k、対照を前後に挟んで 2 回平均。単位は t/s ([results/speed.csv](results/speed.csv))。

| | prefill 4.3k | decode 4.3k | prefill 15k | decode 15k |
|---|---|---|---|---|
| gfx906opt (b9d9b03) | 165 | 27.8 | 153 | 26.2 |
| **cyber-opt1** | **279** | **31.0** | **251** | **29.4** |
| 同上 + MTP 自己投機 (下書き 2) | | 46〜47 | | |

## わかったこと

- **Vulkan に IQ4_XS の整数 dot 経路が無かった。** prefill は f16 に戻して float の matmul を回していた。
  MMQ と MMVQ を足し、IQ4_XS と Q5_K の MMQ を GCN 用の `_int` タイルに載せて prefill +69%、decode +12%。
- **代償: 貪欲出力が対照と 3 トークン目で分かれる。** 活性化を q8_1 に量子化するため。
  KLD は平均 0.0004 で基準内 ([results/kld.csv](results/kld.csv))。同じビルドでは再起動を跨いで一致。採否は未決定。
- **MTP 自己投機は下書き 2 が最良。** +45〜51% で出力は投機なしと一致。下書き 3 は速いが新規コードで出力が分かれた
  ([results/mtp_self_speculation.csv](results/mtp_self_speculation.csv))。
- **Bonsai で効いた FA GQA タイル、小さい m の大きい WG、GDN 16 レーンは効かなかった** (外したほうが同等か速い)。
  GDN 状態のその場書き戻しは 15k decode +1.9%。
- **16GB にはぎりぎり。** 最大 ctx は KV q8_0 で 49152、f16 で 16384、MTP 投機ありで 16384
  ([results/max_ctx.csv](results/max_ctx.csv))。

改造ブランチ cyber-opt1 (gfx906opt b9d9b03 起点) は推論ホストに入れなくなったため、まだこの repo に入っていない。
