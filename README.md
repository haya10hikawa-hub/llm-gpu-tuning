# dirOllamaSetting — Radeon VII (gfx906) 推論最適化

Qwen3.8-27B を AMD Radeon VII (Vega 20 / gfx906、行列演算ユニット非搭載) で
高速に動かすための計測・最適化一式。

**完全な最適化台帳は [OPTIMIZATION.md](OPTIMIZATION.md)。**
採用した施策、棄却した施策とその理由、測定手法の落とし穴、未解決項目を記録している。

## 現在の最良構成

```bash
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 \
  ./work/llama.cpp/build/bin/llama-server \
    -m models27b/Qwen3.8-27B-UD-Q4_K_S.gguf -ngl 99
```

| | 当初 | 現在 |
|---|---|---|
| decode | 13.27 t/s | **26.26 t/s** (+98%) |
| prefill | 98.96 t/s | **173.19 t/s** (+75%) |

`GGML_VK_DISABLE_*` はこれ以外設定しないこと。特に `DISABLE_INTEGER_DOT_PRODUCT`
と `DISABLE_MMVQ` は大きな性能低下を招く (OPTIMIZATION.md の A-02 / A-03)。

## ビルド

```bash
sudo apt-get install -y build-essential cmake git libvulkan-dev glslc \
                        glslang-tools spirv-headers libcurl4-openssl-dev
cd work/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

llama.cpp は upstream `3cf0325` + ローカルパッチ `f48049e`
(GCN で quant matmul の累算を f32 にする、7行)。

## ファイル構成

| パス | 内容 |
|---|---|
| `OPTIMIZATION.md` | **最適化台帳**。採用/棄却の全項目と根拠 |
| `stage0_predictions.md` | Stage 0 の事前予測 (実行前に記録したもの) |
| `gpuclk.sh` | GPU クロック/DPM 制御と実測表示 |
| `run_bench.py` | 計測基盤。クロックを同時サンプリングする |
| `bench_stages.py` | 段階マトリクス (クロック / env / 量子化 x env / op別) |
| `bench_greedy.py` | 貪欲探索 (提出粒度 / CLI グリッド / opプロファイル / 投機デコード) |
| `stage0_sweep.sh` | env スイープ。対照挟み込みと GPU 使用検証つき |
| `quality_eval.sh` | 量子化形式ごとの perplexity |
| `fetch_models.sh` / `fetch_models_27b.sh` | GGUF 取得 |
| `results/` | 8B の測定結果 (CSV / op プロファイル) |
| `results27b/` | 27B の測定結果 |
| `models/` `models27b/` | GGUF (git 管理外) |
| `work/llama.cpp/` | ビルド済み llama.cpp (git 管理外) |

## 測定するときの必須事項

1. **GPU ジョブは常に1つ。** 同時実行すると VRAM 枯渇 → GPU リセット → DRM 権限喪失 →
   無言で CPU 実行に落ちる
2. **`no usable GPU` を毎回検査する。** CPU フォールバックは警告だけで進行する
3. **対照構成を 3 run ごとに挟む。** 連続稼働で VRAM が断片化し decode が 40% 落ちる
4. **`llama-bench` の tg 値と `test-backend-ops` の GFLOPS は使わない。**
   それぞれ 2.8 倍の乖離、PCIe 律速という問題がある

詳細は OPTIMIZATION.md の 5 章・6 章。

## 復旧手順

GPU が見えなくなったとき (`ggml_vulkan: No devices found`):

```bash
sudo setfacl -m u:$USER:rw /dev/dri/renderD128 /dev/dri/card1
sudo usermod -aG render,video $USER
vulkaninfo --summary | grep deviceName   # 復帰確認
```
