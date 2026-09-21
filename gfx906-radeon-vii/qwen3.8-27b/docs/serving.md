# OpenAI 互換 API としての常駐

`llama-server` は元から OpenAI 互換。実作業は**常駐化・外部バインド・認証**の 3 つ。

## 構成 (実測で決定)

```
モデル   Qwen3.8-27B-UD-Q4_K_S (15.36 GB)
ctx      32768 (--parallel 1 = 1 リクエストで全量を使える)
KV       q8_0  (k/v とも)
batch    --batch-size 512 --ubatch-size 256
その他   -ngl 99 --jinja --metrics
bind     0.0.0.0:8080 + --api-key + ufw で送信元を限定
```

`--parallel 1` は長文向けの選択。2 にすると総スループットが 1.48 倍になる代わりに
1 リクエストあたり 16384 トークンに半減する。用途で決める。

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

`Restart=always` にすること。`on-failure` だと**正常終了で再起動しない**。
実際に 3 時間稼働後に status=0 で終了し、常駐が途切れた。

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

## ファイアウォール

`0.0.0.0` バインドのままでは LAN 全体に届く。**ufw で送信元を明示的に限定する。**
設定は `deploy/ufw-rules.sh`。

```
既定          incoming=deny / outgoing=allow
22/tcp        192.168.3.0/24, 10.96.71.0/24 のみ
8080/tcp      192.168.3.98, 10.96.71.67 のみ
9993/udp      Anywhere (ZeroTier は NAT 越えのため任意の対向から受ける必要がある)
```

**適用順序が重要。** SSH の許可を入れる前に `ufw enable` すると、実行中の SSH
セッションごと切れる。本作業時は 4 本の SSH が張られていた。

ufw には `22/tcp ALLOW IN Anywhere` が既存で入っていることがある。有効化後に
削除しないと SSH が全世界に開いたままになる。

## セキュリティ

- キーは `/etc/llm-api.env` (0600) に置き、リポジトリにはテンプレートのみ
- 防御は ufw の許可リストと API キーの 2 層。`--host 0.0.0.0` のままなので、
  **ufw が落ちると LAN 全体に露出する**
- LAN 外に出すならリバースプロキシ + TLS を前段に置く
- 許可先の追加は `sudo ufw allow from <ip> to any port 8080 proto tcp`
