# MiMo-V2.6-Distill-Qwen-9B Q6_K: 単発速度記録

対象GGUFを `/props.model_path` で照合し、32k context、parallel=1、KV K/V Q8、cacheなしで測定した。

| phase | tokens / ms | 速度 |
|---|---:|---:|
| prefill | 2019 / 3672.005 | **549.84 t/s** |
| decode | 230 / 3868.517 | **59.20 t/s** |

入力は固定合成入力、各点1回。cache_n=0は確認したが、反復性・品質・持続負荷は未検証。3回クリーン測定を別途実行中。
