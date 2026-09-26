# パッチ一覧と効果

[../patches/](../patches/) は PrismML フォーク 842b188 に `git am` で当たる。
どれも環境変数 `GGML_VK_DISABLE_*` で個別に切れる。数値は decode 4.3k (t/s)。

| 段階 | 主な変更 | decode | prefill |
|---|---|---|---|
| 環境変数のみ | | 27.6 | 100 |
| opt7 | FWHT を 1 行 256 スレッドに、PQ2_0 mat-vec の活性化を並べ替え、PQ2_0 の MMQ を有効化 | 37.1 | 232 |
| opt8 | GDN ゲートを GDN 内で計算など (起動 −288 回) | 37.3 | 241 |
| opt9 | RMS_NORM と FWHT の融合 (起動 −129 回) | 37.3 | 241 |
| opt10 | FWHT が q8_1 も書く、GDN 状態のその場書き戻し、畳み込みの 1 本化、ほか | 43.8 | 244 |
| opt11 (`RM_KQ_INT=4`) | PQ2_0 mat-vec のアドレス計算を前計算 | 43.9 | 243 |
| opt11 + KV f16 | `-ctk f16 -ctv f16` と `GGML_VK_RM_KQ_INT=2` | **47.2** | **250** |

## 学んだこと

**1. 効くのは起動回数ではなく同期の回数。**
opt8・opt9 は起動を 417 回減らしたが速度は変わらなかった。効いたのは
依存の段を消す変更で、同期 1 回あたり約 5 µs
([results/barriers.csv](../results/barriers.csv))。

**2. 三値 mat-vec はメモリ律速。**
ISA で見ると命令の半分がアドレス計算だったので削ったが (dot4 あたり 4.5 → 3.1 命令)、
速度は +3〜5% だった ([results/pq2_matvec_isa.csv](../results/pq2_matvec_isa.csv))。
残りの差は同時に出せる読み込みの数と見ている。

**3. CPU 側にも穴があった。**
presence penalty が語彙 24.8 万件すべてを毎トークン探索していた (2.9 ms)。
罰則対象だけを直接更新して 0.001 ms、ロジットは完全一致
([results/sampler_cpu.csv](../results/sampler_cpu.csv))。本番の Qwen3.8 にも効くはず。

**4. MTP ヘッドによる自己投機は新規コードでも効く。**
MTP ヘッド付き GGUF で下書き 3 トークンにすると、改変前ビルドで新規コード +49%、
書き換え +70% ([results/mtp_self_speculation.csv](../results/mtp_self_speculation.csv))。
ngram や 0.8B 下書きモデルによる投機は新規コードで逆効果だった。高速化ビルドとの組み合わせは計測待ち。
