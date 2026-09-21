# セットアップ

## 依存

```bash
sudo apt-get install -y build-essential cmake git libvulkan-dev glslc \
                        glslang-tools spirv-headers libcurl4-openssl-dev vulkan-tools
```

`spirv-headers` が無いと Vulkan バックエンドの cmake が失敗する。

## llama.cpp

upstream `3cf0325` + `gfx906-radeon-vii/patches/gcn-f16acc.patch`。

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git work/llama.cpp
cd work/llama.cpp
git apply ../../gfx906-radeon-vii/patches/gcn-f16acc.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

起動時に次が出れば正しく gfx906 を掴んでいる。

```
0 = AMD Radeon VII (RADV VEGA20) (radv) | warp size: 64 | int dot: 1 | matrix cores: none
```

## モデル

```bash
./fetch_models_27b.sh     # Qwen3.8-27B 6形式 (63GB) -> models/
```

推奨は `Qwen3.8-27B-UD-Q4_K_S.gguf` (15.36GB)。
長文プロンプト中心なら bartowski の `Qwen3.8-27B-Q3_K_L.gguf` (14.12GB)。

## 実行

```bash
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 \
  ./work/llama.cpp/build/bin/llama-server \
    -m models/Qwen3.8-27B-UD-Q4_K_S.gguf \
    -ngl 99 -c 24576 --parallel 2 --jinja
```

`--jinja` は tool calling に必要。`-c` は 27k が VRAM 上限。

## 計測

```bash
../gpuclk.sh high                  # クロック固定 (計測時のみ)
./stage0_sweep.sh                 # 環境変数スイープ (対照挟み込みつき)
./quality_eval.sh 40              # perplexity
./practical_sustained.sh 1500     # 持続負荷 + 温度追跡
python3 ../../tools/agentic_eval.py  --server http://127.0.0.1:11439 --out results/a1.csv
python3 ../../tools/agentic_eval2.py --server http://127.0.0.1:11439 --out results/a2.csv
python3 ../../tools/agentic_eval3.py --server http://127.0.0.1:11439 --out results/a3.csv
../gpuclk.sh auto                  # 復帰
```

測定時は `../docs/measurement.md` の規約に従うこと。
