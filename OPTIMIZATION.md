# gfx906 (Radeon VII) LLM 推論最適化 — 台帳

対象: Qwen3.8-27B (`general.architecture = qwen35`) / AMD Radeon VII (Vega 20, gfx906)
記録開始: 2026-09-20 / 最終更新: 2026-09-21

---

## 1. 到達点

| | 当初 (報告値) | 現在 | 改善 |
|---|---|---|---|
| decode | 13.27 t/s | **26.26 t/s** | **+98%** |
| prefill | 98.96 t/s | **173.19 t/s** | **+75%** |

当初: Windows + Ollama + Vulkan + `Qwen3.8-27B-UD-Q3_K_XL`
現在: Ubuntu 26.04 + llama.cpp (patched) + `Qwen3.8-27B-UD-Q4_K_S`

### 推奨構成

```bash
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 \
  llama-server -m Qwen3.8-27B-UD-Q4_K_S.gguf -ngl 99
```

`GGML_VK_DISABLE_*` はこれ以外一切設定しないこと (下記 A-02/A-03 参照)。
llama.cpp は commit `3cf0325` + パッチ `f48049e` でビルドしたもの。

---

## 2. ハードウェア/環境の確定事実 (実機確認済み)

```
GPU        : Vega 20 [1002:66AF] Radeon VII, 16 GB HBM2 (4096-bit, 1024 GB/s)
llama.cpp  : matrix cores: none | warp size: 64 | int dot: 1 | fp16: 1 | shared mem: 65536
coopmat    : 非対応 (llvmpipe のみ公開)
subgroup   : min = max = 64 (wave64 固定、wave32 不可)
int dot    : integerDotProduct4x8BitPackedSignedAccelerated = true
PCIe       : Gen3 x16 (8.0 GT/s), 実測 10.4 GB/s
DPM        : mclk {350, 800, 1000} MHz / sclk {700..1801} MHz
driver     : Mesa RADV 26.0.8 / kernel 7.0.0-30 / Ubuntu 26.04 (Live USB + persistence)
ROCm       : 未導入。gfx906 は ROCm 5.7 でサポート打ち切り。Vulkan が唯一の経路
```

### モデル構造

```
qwen35.block_count             = 65   (線形アテンション 49層 + full attention 16層)
qwen35.full_attention_interval = 4
qwen35.embedding_length        = 5120 / feed_forward = 17408
qwen35.attention.head_count    = 24 / head_count_kv = 4 / key=value_length = 256
qwen35.ssm.state_size = 128 / inner_size = 6144 / time_step_rank = 48 / conv_kernel = 4
qwen35.nextn_predict_layers    = 1    (MTP ヘッドあり)
再帰状態: ssm 3.15 MB + conv 0.12 MB / 層 → 49層で 160 MB
```

---

## 3. 採用した最適化 (ACCEPTED)

| ID | 項目 | 方法 | 効果 | 証拠 |
|---|---|---|---|---|
| A-01 | Ollama → llama.cpp 直接 | native ビルド | +74% | 同一モデル・同一GPU で Ollama 8.4 / llama.cpp 14.6 (※断片化状態での比較) |
| A-02 | `DISABLE_INTEGER_DOT_PRODUCT` を**設定しない** | env 除去 | prefill +30% | 27B: 132.9 → 102.1 (切ると -23%) / 8B: 742.8 → 360.9 (-51%) |
| A-03 | `DISABLE_MMVQ` を**設定しない** | env 除去 | decode +13% | 26.28 → 23.27 (切ると -11.5%) |
| A-04 | `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | env 追加 | +20% | Q4_K_S: 21.77 → 26.11。VRAM 余裕が小さいほど効く (Q3_K_XL では +3.6%) |
| A-05 | VRAM に収まる最大の量子化を選ぶ | Q4_K_S (15.36GB) | +7% | Q2_K_XL 24.73 / Q4_K_S 25.85。小さくしても速くならない |
| A-06 | **f16acc パッチ** (`f48049e`) | コード 7行 | prefill +30.3% | 132.90 → 173.19、decode 26.28 → 26.26 (不変) |

### A-06 の詳細

`ggml_vk_get_mul_mat_mat_f16acc()` が gfx906 で f16 累算を選んでいた。GCN には行列ユニットが無く、
ACO がスカラ `float16_t` の累算器をペアに詰めないため、レート利得ゼロで変換コストだけが乗る。

```cpp
// ggml/src/ggml-vulkan/ggml-vulkan.cpp:5702 付近
if (ctx->device->architecture == vk_device_architecture::AMD_GCN) {
    return false;
}
```

機序の実証 (型別 MUL_MAT 比較):

```
iq4_xs  2975.5 → 2071.2 ms  -30.4%   I-quant (MMQ 不可 → f16acc が有効だった)
iq3_xxs  156.1 →  106.9 ms  -31.5%
iq4_nl   137.5 →   95.0 ms  -30.9%
q5_K     824.0 →  833.4 ms   +1.1%   K-quant (MMQ 経路 → f16acc は元から無効)
q4_K     779.6 →  784.6 ms   +0.6%
────────────────────────────────────
I-quant 合計 -28.6% / K-quant 合計 +0.9%
```

`AMD_GCN` は `minSubgroupSize == maxSubgroupSize == 64` で判定される。RDNA 以降は非該当。
Vega 10 / Polaris / Fiji でも同じ利得が期待できる (未検証)。

---

## 4. 棄却した施策 (REJECTED) — 再試行しないこと

| ID | 項目 | 結果 | 棄却理由 |
|---|---|---|---|
| R-01 | GPU クロック固定 (`power_dpm_force_performance_level=high`) | 効果なし | `auto` が負荷時に既に 1000MHz へ昇圧。350MHz 固定では decode -40% (対照として有効) |
| R-02 | `MAX_NODES_PER_SUBMIT` 11水準 (1〜8192) | ±3% | 既定 100 が最適。`=1` で -14%、`SERIALIZE` で -34% (機序は確認) |
| R-03 | flash attn × ubatch × KV量子化 12構成 | ±0.6% | 全構成 8.52〜8.57 (当時の測定系) |
| R-04 | MTP 投機デコード (`--spec-type draft-mtp`) | 最良 +6.6% | n_max=2 で 9.07、4 で 7.06、8 で 3.31。受理率が急落。混合再帰では状態更新が直列で償却できない |
| R-05 | ngram 投機デコード | ドラフト生成ゼロ | n_drafted = 0 |
| R-06 | AMDVLK ドライバ | 起動不可 | kernel 7.0 で `vkCreateInstance` → `ERROR_INCOMPATIBLE_DRIVER (-3)` |
| R-07 | **get_rows ベクトル化** (Level B) | 0% | カーネル時間 150.6 → 156.7 us で不変。正当性 240/240 合格だが無効。**そもそも断片化による人工物を見ていた** |
| R-08 | **再帰状態の in-place 化** (Level A) | 着手せず | クリーン状態では GET_ROWS 1.7% + CPY 1.6% = 3.3% のみ。改修価値なし |
| R-09 | **IQ4_XS 向け q8_1 MMVQ 追加** (Stage 3-A) | 着手せず | decode での I-quant vs K-quant は 455 vs 495 GB/s = 8% 差のみ。期待値 ~4% |
| R-10 | `GGML_VK_FORCE_MMVQ=1` | -1.9% | AMD は `k >= 2048` で既定 true (`ggml-vulkan.cpp:6310`)。no-op |
| R-11 | `GGML_VK_DISABLE_DOT2=1` | -0.2% | `v_dot2_f32_f16` 経路は本件で使われていない |
| R-12 | `ENABLE_MEMORY_PRIORITY` / `ASYNC_USE_TRANSFER_QUEUE` | ±1% | 中立 |
| R-13 | `ALLOW_GRAPHICS_QUEUE=1` | +1.6% | ノイズ範囲 (対照レンジ 1.7%)。8B では +3.8%、27B では -0.1% とモデル依存 |
| R-14 | `DISABLE_FUSION` / `GRAPH_OPTIMIZE` / `MULTI_ADD` | -0.3 ~ -4% | 既定 (有効) が最適 |

---

## 5. 測定手法 — 信頼できるもの / できないもの

### 使ってはいけない

| ツール | 問題 | 実測 |
|---|---|---|
| `llama-bench` の `tg` | 混合再帰アーキテクチャで実生成と乖離 | 8.59 vs 実測 24 (2.8倍) |
| `test-backend-ops perf` の GFLOPS | 23型中15型が PCIe 転送律速 | 19MB〜235MB の 12倍幅で逆算帯域が 10.4 ± 0.2 GB/s に固定 |

`test-backend-ops` の GFLOPS で量子化形式の演算コストを比較することはできない。

### 使うべき

- `llama-completion` / `llama-server` の実生成 (`prompt eval time` / `eval time`)
- `GGML_VK_PERF_LOGGER=1` による op 別 GPU 時間
- `test-backend-ops test` (正当性のみ。perf は使わない)

### 測定規約 (違反すると結果が壊れる)

1. **GPU ジョブは常に1つ**。起動前に `gpu_busy_percent` と `mem_info_vram_used` を確認
2. **全 run で `no usable GPU` を検査**し、該当は破棄
3. **3 run ごとに対照構成を挟む**。ドリフト検出用
4. **変更前に反証可能な予測を書く**。外れたら中断して再診断
5. 初回 run はシェーダコンパイル費用を含むので分けて扱う
6. **効果量の予測は実測値の範囲内に留める。** 外挿するなら外挿と明示する (10章参照)

---

## 6. 環境由来の障害 — 発生パターンと対処

| 事象 | 症状 | 原因 | 対処 |
|---|---|---|---|
| VRAM 断片化 | decode が 24 → 14.5 t/s に劣化 (-40%) | 数時間の連続ベンチでアロケータが断片化し、バッファが遅い領域に落ちる | プロセス再起動。常駐運用では定期再起動 |
| GPU リセット | `dmesg`: `Not enough memory for command submission` / `amdgpu_vm_validate() failed` | 複数プロセスが同時に 13GB 超を確保 | 同時実行しない |
| DRM 権限喪失 | `ggml_vulkan: No devices found` / `no usable GPU` → 無言で CPU 実行 (0.85 t/s) | GPU リセットで `/dev/dri/*` が再生成され ACL が消失 | `sudo setfacl -m u:$USER:rw /dev/dri/renderD128 /dev/dri/card1`、`usermod -aG render,video $USER` |

**CPU フォールバックは警告のみで進行するため、検査しないと性能値として混入する。**

---

## 7. 未解決 / 次の候補

| ID | 項目 | 現状 | 期待値 |
|---|---|---|---|
| O-01 | matvec の実効帯域 | 加重平均 474 GB/s (ピークの 46%)。最速 q5_K 662 GB/s (65%) / 最遅 iq3_s 288 GB/s (28%) | 全形状が 650 GB/s なら decode +26% (→ 33 t/s) |
| ~~O-02~~ | ~~純 K-quant モデル (非 UD)~~ | **完了 → 10章**。decode は棄却 (+4%)、prefill は採用 (+22%) | 用途で使い分け |
| O-03 | `q8_0 m=48 k=5120` | 5472回/96token、実効 16 GB/s (1.6%)、matvec の 5.1% | m=48 では 60 CU を埋められない。split-K が必要。+3〜4% |
| O-04 | Stage 2 (`RADV_DEBUG=shaderstats`/`asm`) | 未実施 | 占有率・レジスタ圧・命令列の直接観測。O-01 の内訳確定に必要 |
| ~~O-05~~ | ~~品質 (perplexity)~~ | **完了 → 9章** | Q4_K_S が速度・品質とも最良。トレードオフなし |

### O-01 の根拠 — 同一形状での型比較

```
m=17408, k=5120
  q5_K   61.3 MB  92.5 us  662 GB/s (65%)
  iq3_s  38.3 MB 132.9 us  288 GB/s (28%)
```

iq3_s は 37% 少ないバイトしか読まないのに 44% 長い。**帯域では説明できず、逆量子化 ALU が律速**。
これは当初アーキテクチャ提案の中核仮説「逆量子化コストが到達可能な帯域の上限を決める」の直接証明。

---

## 8. 誤診の記録 — 同じ轍を踏まないために

| 誤った診断 | 実際 | 見落とした点 |
|---|---|---|
| mclk が 350MHz に張り付く | `auto` で正しく昇圧 | アイドル値を負荷時の値と誤認 |
| 環境変数の綴り誤りが原因 | 転記ミスで実環境は正しい | 本人に確認せず断定 |
| Ollama が GPU を使えていない | Vulkan で正しく動作 | ROCm 非対応の事実から飛躍 |
| I-quant が gfx906 で優位 | `DISABLE_INTEGER_DOT_PRODUCT` が作った人工物 | 比較条件が揃っていなかった |
| 線形アテンションの再帰演算が律速 | `GATED_DELTA_NET` は 1.1% | op 名から推測した |
| 状態の読み書きが 36.6% | **クリーン状態では 3.3%** | **VRAM 断片化した環境で測定していた** |
| カーネルのスレッド粒度が主因 | 無関係 | 上記の人工物を構造的欠陥と誤認 |
| `DISABLE_HOST_VISIBLE_VIDMEM` で +72% | +3.6〜20% | 劣化した baseline と比較していた |
| 残る余地は 2倍 | +20〜40% | matvec 以外の時間を分母に含めて帯域を過小計算 |

**共通の失敗要因: 測定環境の異常を対象の性質と取り違えた。**
規約 #3 (対照の挟み込み) と #4 (事前予測) はこの再発防止のために導入した。

---

## 9. 品質 — 量子化による劣化率 (O-05, 完了)

測定: `llama-perplexity`, wikitext-2 test, `-c 512 --chunks 40`, 全層 GPU

| 形式 | サイズ | PPL | ± | 劣化率 | decode t/s |
|---|---|---|---|---|---|
| UD-Q2_K_XL |  9.83 GB | 6.2641 | 0.14881 | **+5.55%** | 24.73 |
| UD-IQ3_XXS | 10.93 GB | 6.1265 | 0.14583 | **+3.23%** | 23.99 |
| UD-Q3_K_XL | 13.15 GB | 6.0266 | 0.14385 | **+1.54%** | 24.06 |
| UD-IQ4_XS  | 14.25 GB | 5.9474 | 0.14113 | **+0.21%** | 24.56 |
| **UD-Q4_K_S** | **15.36 GB** | **5.9350** | 0.14073 | **0.00%** | **25.85** |

### 結論

**Q4_K_S が速度・品質の両方で最良。トレードオフは存在しない。**

- 14 GB を超えると劣化曲線は平坦化する (14.25 → 15.36 GB で品質改善は 0.21% のみ)
- 13 GB を下回ると劣化が加速する (13.15GB で +1.54%、9.83GB で +5.55%)
- **サイズを削っても decode は速くならない** (23.99 〜 25.85 で横ばい、最大モデルが最速)
- したがって小さい量子化を選ぶ理由は「VRAM に載らない」以外に無い

これは A-05 を速度・品質の両面から確定させるもの。

### 留保

- 40 チャンクでの測定のため誤差は ±2.4%。隣接形式間の差 (0.2〜1.7%) は個別の
  誤差範囲内だが、同一コーパス・同一チャンクで誤差が相関するため相対比較は有意。
  **5 点すべてが単調である**こと自体が証拠になる
- 完全精度 (BF16 50GB) のリファレンスは VRAM に載らないため、絶対的な劣化率は不明。
  上表は Q4_K_S を基準とした相対値
- perplexity は下流タスク性能の代理指標にすぎない。実用性の評価には別途ベンチマークが必要
- モデル自体の能力 (Qwen3.8-27B が何をどこまでできるか) は未評価

---

## 10. 量子化型と性能の正しいモデル (O-02, 完了)

### 対照実験

ほぼ同サイズの 2 モデルで量子化型の構成だけを変えた (サイズ差 0.9%)。

| モデル | サイズ | prefill | decode | PPL |
|---|---|---|---|---|
| unsloth `UD-IQ4_XS` (I-quant 混在) | 14.25 GB | 170.8 | 24.89 | 5.9474 |
| bartowski `Q3_K_L` (純 K-quant) | 14.12 GB | **208.7 (+22.2%)** | 25.91 (+4.1%) | 6.0442 (+1.63%) |

### matvec の型別実効帯域 (decode)

```
Q3_K_L  (純K)   総 1339.2 ms  転送 579.35 GB  加重平均 432.6 GB/s
  q3_K=353GB/s(57%)  q4_K=618GB/s(22%)  q5_K=513GB/s(17%)

UD-Q4_K_S (混在) 総 1738.9 ms  転送 824.08 GB  加重平均 473.9 GB/s
  iq4_xs=489GB/s(42%)  q4_K=602GB/s(16%)  q5_K=526GB/s(16%)  iq3_s=287GB/s(7%)
```

**純 K-quant の方が加重平均は 9% 低い。** decode が +4% 速かったのは帯域ではなく
転送量が 30% 少ないため。

### 正しいモデル — 2 つの独立した軸

型別帯域を bits-per-weight で並べると説明がつく。

```
q3_K   3.44 bpw -> 353 GB/s        iq3_s  3.44 bpw -> 287 GB/s
iq4_xs 4.25 bpw -> 489 GB/s        q4_K   4.50 bpw -> 618 GB/s
q5_K   5.50 bpw -> 513 GB/s
```

| 軸 | 影響先 | 内容 |
|---|---|---|
| **bits-per-weight** | **decode** | 低ビットほど逆量子化コストが相対的に重く到達帯域が下がる。バイト削減と相殺するため decode はサイズにほぼ不感 |
| **MMQ 経路の可否 (K vs I)** | **prefill** | I-quant は q8_1 + `v_dot4_i32_i8` の GEMM 経路に乗れない。純 K-quant で +22% |

当初アーキテクチャ提案の「逆量子化コストが到達可能な帯域の上限を決める」は正しい。
ただし **軸は K/I の別ではなく bpw**。「I-quant が遅い」という括り方は誤りで、
q3_K(353) と iq3_s(287) は同程度に遅く、iq4_xs(489) は q4_K(618) に迫る。

### 推奨 — 用途で分かれる

| 用途 | モデル | 根拠 |
|---|---|---|
| 長文プロンプト / RAG / 文書処理 | `Q3_K_L` 14.12 GB | prefill 208.7 (+20.5%)、VRAM 余裕 2.2 GB |
| チャット / 短文入力・長文生成 | `UD-Q4_K_S` 15.36 GB | decode 26.26、PPL 5.9350 (最良) |

16 GB に収まる純 K-quant は `Q3_K_L` (14.12 GB) が最大。
`Q4_K_S` (16.36 GB) と `Q3_K_XL` (16.39 GB) はいずれも VRAM 超過。

### 予測精度についての教訓

| 予測 | 結果 |
|---|---|
| 機序 (I-quant のみ f16acc の影響を受ける) | 的中 |
| 機序 (純 K-quant で prefill 向上) | 的中 |
| 効果量 (decode +15 ~ 22%) | **外れ (+4.1%)** |
| 効果量 (f16acc 適用後 prefill 149 ~ 150) | **外れ (173)** |

**機序の予測は当たるが、効果量の予測は外れる。** 原因はいずれも
「実測値からの外挿」。今回も Stage 1 の「I/K で 8% 差」という実測を持ちながら
「加重平均が 474 -> 620 に跳ぶ」と過大に外挿した。

→ 規約 #6 として追加: **効果量を予測するときは実測値の範囲内に留め、
外挿する場合は外挿であることを明示する。**

---

## 11. 律速要因の確定 (O-04, 完了)

外部観測 (クロック感度) と内部観測 (パイプライン統計) の両方で一致した。

### 外部観測 — クロック感度

sclk と mclk を独立に振り、decode の弾性を測定 (`Q3_K_L`, 基準 26.2 t/s)。

```
mclk 1000 -> 800  (-20%)   26.17 -> 25.01  (-4.4%)   弾性 0.22
mclk 1000 -> 350  (-65%)   26.17 -> 14.31 (-45.3%)   弾性 0.70
sclk 1801 -> 1683 (-6.6%)  26.23 -> 25.50  (-2.8%)   弾性 0.42
sclk 1801 -> 1372 (-23.8%) 26.23 -> 21.89 (-16.5%)   弾性 0.69
sclk 1801 ->  808 (-55.1%) 26.23 -> 13.95 (-46.8%)   弾性 0.85
```

**動作点近傍で sclk 弾性 0.42 / mclk 弾性 0.22。演算側が約 1.9 倍効く。**

注: 大きな摂動では両方とも弾性 0.7-0.85 に近づく。どちらの資源も落としきれば
律速側に回るため、判定に使えるのは小摂動の弾性のみ。

### 内部観測 — パイプライン統計

`GGML_VK_PIPELINE_STATS=mul_mat_vec` + `test-backend-ops test -o MUL_MAT`

| type | VGPR | waves/SIMD | code size | LDS | 実測 GB/s |
|---|---|---|---|---|---|
| q6_k | 256 | **1** | 41864 | 512 | 343 |
| iq4_xs | 256 | **1** | 88672 | 512 | 489 |
| iq3_s | 256 | **1** | 51396 | 2048 | 287 |
| iq3_xxs | 256 | **1** | 50264 | 1024 | 346 |
| iq2_xs | 256 | **1** | 50128 | 4096 | - |
| q3_k | 128 | **2** | 17132 | 512 | 353 |
| **q4_k** | **64** | **4** | 6732 | 0 | **618** |
| **q5_k** | **64** | **4** | 7760 | 0 | **513** |
| q8_0 | 36 | 7 | 9276 | 0 | 16 (m=48 の病理、別要因) |

### 診断

**真因は「ALU 命令数」ではなく「レジスタ圧による occupancy 低下 -> メモリレイテンシの露出」。**

GCN の SIMD はレーンあたり 256 VGPR。256 使えば 1 wave しか走らず、
HBM のレイテンシ (400-800 cycle) を隠せない。

これは **wave64 固有の問題**:

```
gfx906 (GCN,  wave64) : SIMD あたり 256 VGPR -> 256 VGPR 使用で 1 wave
RDNA   (      wave32) : SIMD あたり 512 VGPR -> 同じシェーダで 4 wave
```

シェーダが RDNA のレジスタファイルを前提にチューニングされているため、
wave64 の gfx906 では occupancy が 1/4 に落ちる。
「最新の専用最適化パスほど gfx906 で遅い」という当初の報告の機序がこれ。

### 留保

occupancy だけでは説明しきれない。`iq4_xs` は 1 wave でも 489 GB/s を出し、
2 wave の `q3_k` (353) を上回る。相関はあるが決定的因子ではない。
wave 内のメモリレベル並列度 (outstanding load 数) も効いていると考えられる。

### O-01 の標的 (更新)

**「ALU を減らす」ではなく「VGPR を減らして wave 数を稼ぐ」。**
目標は `q4_K`/`q5_K` と同水準 (VGPR <= 64、LDS 0)。

期待値 (規約 #6 に従い実測範囲内): q4_K が実測 618 GB/s を達成しているので、
他型がそこに近づけば `Q3_K_L` の加重平均 432 -> 600 前後、matvec 時間 -28%、
decode +25% 程度。ただし iq4_xs の例外があるため確実ではない。

---

## 12. 運用上の追加候補

| 項目 | 根拠 | 状態 |
|---|---|---|
| サーマルスロットリングの影響 | decode は sclk 弾性 0.42。sclk が 10% 落ちれば性能 4% 低下。Radeon VII は junction 110C で絞られる | 未測定 |

### 作業上の注意 (本セッションで 4 回事故)

`pgrep -f` / `pkill -f` のパターンが**自分のコマンドライン**にマッチして自プロセスを
kill する。echo の文字列にもマッチするので注意。ブラケット表記で回避:

```bash
pgrep -f "bin/llama-comp[l]etion"    # OK
pgrep -f "bin/llama-completion"      # 自滅する
```

`llama-completion` は生成後に対話モードへ移行して stdin を待つため、
バッチ測定では **`-st` (single-turn) と `< /dev/null` を必ず付ける**。
付け忘れると timeout で殺され perf 出力に到達せず、空値になる。
