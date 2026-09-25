# MiMo-V2.6-Distill-Qwen-9B Q4_K_M: 単発速度記録

対象GGUFを `/props.model_path` で照合し、32k context、parallel=1、KV K/V Q8、cacheなしで測定した。

| phase | tokens / ms | 速度 |
|---|---:|---:|
| prefill | 2019 / 3465.066 | **582.67 t/s** |
| decode | 230 / 3268.754 | **70.06 t/s** |

prefill要求の付随生成はlength終了だが、prefill timingは2019入力トークンの処理値である。固定合成入力の単発値なので品質推論には使わない。3回クリーン測定は進行中。
