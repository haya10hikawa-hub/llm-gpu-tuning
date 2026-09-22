# MiMo-V2.6-Distill-Qwen-9B Q5_K_M: 単発速度記録

対象GGUFを `/props.model_path` で照合し、32k context、parallel=1、KV K/V Q8、cacheなしで測定した。

| phase | tokens / ms | 速度 |
|---|---:|---:|
| prefill | 2019 / 3710.558 | **544.12 t/s** |
| decode | 230 / 3069.285 | **74.61 t/s** |

入力は固定合成入力、各点1回。cache_n=0であっても、単発値は量子化順位を確定しない。仕様書全体を使う新規サーバー3反復は進行中。
