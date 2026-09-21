# OpenAI 互換 API としての常駐

`llama-server` は元から OpenAI 互換。実作業は**常駐化・外部バインド・認証**の 3 つ。

## 構成 (実測で決定)

```
モデル   Qwen3.8-27B-UD-Q4_K_S (15.36 GB)
ctx      32768 (--parallel 2 で 16384/slot)
KV       q8_0  (k/v とも)
その他   -ngl 99 --jinja --metrics
bind     0.0.0.0:8080 + --api-key
```

### KV を q8_0 にした理由

`-c 24576` を f16 KV で通すと **VRAM 余裕が 28 MB** しか残らない。q8_0 なら
コンテキストを 33% 増やしてなお余裕が 17 倍になる。**速度は同じ。**

| | -c 24576 / f16 | **-c 32768 / q8_0** |
|---|---|---|
| decode | 24.92 / 24.93 | 24.67 / 24.99 |
| VRAM | 16340 MB (余裕 28 MB) | **15898 MB (余裕 470 MB)** |
| slot 毎 ctx | 12288 | **16384** |

品質への影響は未測定。KV の q8_0 量子化は一般にほぼ無損失とされるが、
本機での検証はしていない。

## 導入

```bash
sudo cp deploy/llm-api.env.example /etc/llm-api.env
sudo chmod 600 /etc/llm-api.env
# 生成したキーを書き込む
sudo cp deploy/llm-api.service /etc/systemd/system/
# ExecStart のパスを実環境に合わせる
sudo systemctl daemon-reload && sudo systemctl enable --now llm-api
```

`SupplementaryGroups=render video` は必須。無いと GPU を掴めず**無言で CPU に
落ちる** (27B で約 0.85 t/s)。

起動は **63〜69 秒**。`TimeoutStartSec=300` を確保している。

## 動作確認

```bash
KEY=$(sudo grep LLAMA_API_KEY /etc/llm-api.env | cut -d= -f2)
curl -H "Authorization: Bearer $KEY" http://<host>:8080/v1/models
```

| 確認項目 | 結果 |
|---|---|
| 認証 | キー無し / 誤りとも **401**、正しいキーで 200 |
| `/v1/models` | `qwen3.8-27b` |
| `/v1/chat/completions` | content と `reasoning_content` を分離して返す |
| `/v1/completions` | 動作 (legacy) |
| ストリーミング | SSE。reasoning → content の順に delta |
| tool calling | 動作 (`--jinja` 必須) |
| `/metrics` | 動作 |
| 再起動 | 69 秒で復帰。boot 時自動起動 |

## 推論トークンの制御

通常チャットでは `reasoning_content` が出る。**"OK" の一言に 27 トークン**消費した。

リクエスト単位で切れる:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

これで同じ応答が **2 トークン**になる。サーバ全体で切るなら `--reasoning off`。

**tool calling では元から推論を出さない** (48 タスクで thinking 平均 0)。
エージェント用途では制御不要。

## セキュリティ

`0.0.0.0` バインドは **LAN 全体に公開**する。本機は `ufw` 無効・iptables 全許可
なので、API キーだけが防御線になる。

- キーは `/etc/llm-api.env` (0600) に置き、リポジトリには入れない
- LAN 外に出すならリバースプロキシ + TLS を前段に置く
- 公開範囲を絞るなら `--host` を特定 IP にするか firewall で 8080 を制限する
