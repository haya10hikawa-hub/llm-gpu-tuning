# ステージ実測記録

計画: `~/.claude/plans/immutable-plotting-dahl.md`。規約は repo ルートの `AGENTS.md`。

## 予測と結果

| 予測 | 結果 | 判定 |
|---|---|---|
| N1 公開 API への cold TLS ハンドシェイクは 20ms 超 | 19.93–23.72ms、初回バイトまで 27.89–34.26ms | 的中 |
| N1b warm keep-alive の往復は 5ms 超 | **9.93–10.38ms** (n=4) | 的中 |
| S2 ルーティング判定 p99 < 1ms | **0.73us** | 的中 |
| S2 p99 < 200us (外挿と明記) | 0.73us | 的中 |

## S1 — dirty state の継承機構 (20/20)

一時 index に `git add -A` して `write-tree` + `commit-tree` で seed commit を作る方式。

```
親の (a) 追跡ファイル変更 / (b) staged / (c) 未追跡  → 子に byte 一致で伝播
親の (d) gitignore 対象                          → seed tree に不在 (共有側で扱う)
親の HEAD / status / .git/index mtime / stash ref → すべて不変
孫の seed は .agents/ を除外                      → 再帰安全
3way patch で子の成果を親の dirty tree へ還流      → 親の作業中状態は保持
```

**`git stash create` を棄却した数値:** git 2.53.0 の man は `git stash create [<message>]`。
`-u` は "only valid for push and save commands" と明記。未追跡ファイルが黙って落ちる。

## S1b — 対照と衝突経路 (6/6)

- **R5 解決 (対照つき):** worktree ごとの `info/exclude` は**効かない**。
  対照として `$GIT_COMMON_DIR/info/exclude` は効き、linked worktree にも届く。
  → `.gitignore` か共通 dir のみが正当な退避先。
- **衝突時:** `git apply --3way --check` が正しく拒否し、親のファイルは byte 一致のまま。
  衝突マーカーも `.rej` も残らない。
  なお素の `--3way` もこの例では部分適用しなかったため、`--check` ゲートは
  「必要性を実証済み」ではなく**予防的**。

## S2 — ルーティング判定の遅延 (PASS)

```
n=10000   初回 8.23us (規約 5 により別計上)
p50 0.21us   p90 0.26us   p99 0.73us   max 17.63us
対照 n=3333: p50 0.19us / p99 0.23us  (ドリフトなし)
```

**静的ルール 0.73us 対 ネットワーク 1 往復 ~10,000us = 約 13,700 倍。**
Jev をホットパスのルータに置く案は、ベンダ主張 (~200倍速) を検証するまでもなく
この差で棄却される。律速はネットワーク RTT であって計算ではない。

## 未実施 — モデル重みが不在のため

`models/` `models27b/` `models27b_bartowski/` (約 100GB) がディレクトリごと消失。
`df` は 118G 使用のまま変化していないため、unlink 済みだが fd 保持中の可能性。
再取得は `gfx906-radeon-vii/qwen3.8-27b/fetch_models_27b.sh`。

- **S0** 常駐起動と `n_ctx_slot` の確認 (R1: 12k か 27k か) — 未解決の最大リスク
- **S3** スロット調停と CPU fallback 検出
- **S4** バックエンド間の文脈受け渡し
- **S5** Fugu 段 — API キーなし
