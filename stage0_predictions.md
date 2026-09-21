# Stage 0 事前予測 (実行前に記録)

対象: Qwen3.8-27B-UD-Q4_K_S (15.36 GB), clean state, llama-completion
基準構成 (base): GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 のみ。実測 26.1 t/s

総合予測: **base を 5% 以上上回るノブは無い。**
理由: AMD 経路では k>=2048 で MMVQ が既定で有効 (ggml-vulkan.cpp:6310)、
かつ fusion / graph optimize も既定で有効。既定値が既に最適と考えられる。

| ノブ | 予測 decode | 根拠 |
|---|---|---|
| FORCE_MMVQ=1 | +-2% | AMD k>=2048 で既に true。no-op のはず |
| DISABLE_MMVQ=1 | -10 ~ -20% | K-quant テンソルが v_dot4 経路を失う |
| DISABLE_INTEGER_DOT_PRODUCT=1 | -10 ~ -20% | 同上。prefill はさらに大きく落ちる |
| DISABLE_DOT2=1 | +-2% | 8B で影響なしを確認済み |
| DISABLE_F16=1 | -5 ~ -15% | fp16 ストレージ/演算を失う |
| ALLOW_GRAPHICS_QUEUE=1 | +-2% | 27B の旧測定で -0.1 程度 |
| DISABLE_FUSION=1 | -2 ~ -8% | 融合が効いている分だけ落ちる |
| DISABLE_GRAPH_OPTIMIZE=1 | -2 ~ -8% | 同上 |
| DISABLE_MULTI_ADD=1 | -1 ~ -5% | 同上 |
| MAX_NODES_PER_SUBMIT=2000 | +-2% | 旧測定で +0.6%（ノイズ範囲） |
| DISABLE_ASYNC=1 | -1 ~ -3% | 非同期提出を失う |

## 外れた場合の扱い
予測を外したノブが出たら、そこで一旦停止して原因を特定する。
特に「base を 5% 以上上回るノブ」が出た場合は、Stage 1 以降の前提が変わるため
先にその機序を説明できるまで次に進まない。

## 測定規約
- GPU ジョブは常に1つ
- 全 run で "no usable GPU" を検査、該当は GPUFAIL として破棄
- 3 run ごとに base を挟み、ドリフト(断片化)を検出
- llama-completion のみ使用 (llama-bench tg / test-backend-ops は使わない)
