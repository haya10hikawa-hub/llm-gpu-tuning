# 採用した施策

| # | 施策 | 効果 | 根拠 |
|---|---|---|---|
| 1 | `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | decode 3.6 倍 / prefill +76% | `results/hvv_server.csv` |
| 2 | tool call 文法の改行を任意にするパッチ | 暴走が止まった | `results/agent_mql.csv` |

## 1 — host-visible VRAM

同じバイナリ・同じ深さで、`llama-server` の decode だけが `llama-bench` の
3.7 倍遅かった (116 ms vs 31 ms / token)。起動回数は同じで、差は再帰状態を
読み書きする 3 つの演算に集中していた。

```
            server    llama-bench
GATED_DELTA_NET  27.5 ms   0.86 ms
CPY              25.5 ms   0.73 ms
GET_ROWS         23.7 ms   0.99 ms
```

GATED_DELTA_NET は 1 層あたり約 3 MB の状態を読んで書くので、実効約 11 GB/s。
PCIe の実測帯域 (10.4 GB/s) と同程度。Radeon VII は Resizable BAR 非対応で
CPU からも見える VRAM が 256 MB しかなく、ggml-vulkan はまずここにバッファを
置こうとする。サーバーでは KV (144 MB) と再帰状態 (150 MB) で溢れ、溢れた分が
システム RAM に移されていた。

環境変数でこのヒープを使わせないと、サーバーでも decode 8.74 → 31.38 t/s。
貪欲生成の出力は 714 文字すべて一致した。

## 2 — tool call 文法

llama.cpp の tool call 文法は `</parameter>` の前後に改行がちょうど 1 つ
あることを要求する。モデルが改行を省くと文法が閉じず、出力上限まで生成し
続ける。パッチ前は最初の課題で 16,000 トークン (上限) まで暴走した。

改行を任意にするパッチ ([patches/toolcall-optional-newlines.patch](../../patches/toolcall-optional-newlines.patch))
を当てると、MQL 課題の M1・M2 を全テスト合格で通過した。GPU に依存しない変更。
