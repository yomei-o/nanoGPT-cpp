[English](README.md) | [日本語](README.ja.md)

# nanoGPT — C++ 移植版

Andrej Karpathy の [nanoGPT](https://github.com/karpathy/nanoGPT) を、依存ライブラリなしの
**標準 C++17** だけで移植したものです。GPT-2 の Transformer を **forward だけでなく
backward まで** すべて手書きで実装し、AdamW オプティマイザも自作しているため、
**文字レベル GPT をスクラッチから学習** することも、**本物の GPT-2 事前学習済み重みで
推論** することもできます。

[`model.py`](../model.py)（ネットワーク）と、[`train.py`](../train.py) /
[`sample.py`](../sample.py) の学習・サンプリングのロジックを C++ に移植しています。

> **姉妹プロジェクト:** nanoGPT の後継である nanochat にも、依存なしの C++ 移植版
> [nanochat-cpp](https://github.com/yomei-o/nanochat-cpp) があります（モダンな Transformer:
> RoPE・GQA・Muon・BPE、加えて SFT ＋ chat CLI）。

## Transformer 用の C++ ライブラリは必要？ いいえ。

Transformer は「行列積といくつかの要素ごとの演算」にすぎません。token/位置埋め込み、
LayerNorm、causal マルチヘッド自己注意（QKV の行列積 + softmax）、GELU つき MLP、
残差接続、そして線形ヘッド。これらすべてをここではスクラッチで実装しています。
PyTorch も Eigen も BLAS も ggml/llama.cpp も使わず、**C++ 標準ライブラリのみ**
（Karpathy 自身の llm.c / llama2.c と同じ方針）。OpenMP は存在すれば自動で使いますが、
必須ではありません。

- **MSVC（Visual Studio 2022）、GCC、Clang** で、Windows / Linux / macOS 上で
  そのままビルド・実行できます。
- backprop は数値微分と照合して検証済み。BPE トークナイザは実際の GPT-2 トークン ID と
  照合して検証済み。文字レベル学習も収束を確認済みです。

## 実装内容

`model.py` とまったく同じ GPT-2 アーキテクチャを forward / backward の両方で実装:

- token + 位置埋め込み（重み共有: `lm_head` が `wte` を共有）
- LayerNorm（bias はオプション）
- causal マルチヘッド自己注意
- GELU つき MLP（厳密な `erf`、または GPT-2 が学習に使った `tanh` 近似）
- 残差接続
- クロスエントロピー損失
- decoupled weight decay つき AdamW（nanoGPT と同様、2D テンソルにのみ decay を適用）

## ファイル一覧

| ファイル | 役割 |
|---|---|
| `gpt.h` | GPT モデル本体: 全演算の forward+backward と AdamW（コア） |
| `tokenizer.h` | 文字レベルトークナイザ（学習用） |
| `bpe.h` | GPT-2 のバイトレベル BPE トークナイザ（GPT-2 推論用） |
| `main.cpp` | 文字レベルモデルの `train` / `sample` ドライバ |
| `gpt2.cpp` | 本物の GPT-2 重みを読み込みテキスト生成 |
| `export_gpt2_safetensors.py` | GPT-2 重みを C++ 形式へエクスポート — **numpy のみ、torch 不要**（推奨） |
| `export_gpt2.py` | 同上（nanoGPT の `model.py` 経由。torch + transformers が必要） |
| `test_grad.cpp` | 勾配チェック（倍精度）— backprop の正しさ |
| `bpe_test.cpp` | BPE 出力を既知の GPT-2 トークン ID と照合 |
| `CMakeLists.txt` | クロスプラットフォームビルド（OpenMP を自動検出） |

## ビルド

### Visual Studio 2022（CMake）

Visual Studio 2022 →**ローカル フォルダーを開く**→ この `cpp` フォルダを選択。
VS が `CMakeLists.txt` を自動検出するので、**`x64-Release`** を選び
**ビルド → すべてビルド**。`nanogpt`、`gpt2`、`nanogpt_gradcheck`、`bpe_test`
が生成されます。

### CMake（任意のコンパイラ）

```bash
cmake -S . -B build
cmake --build build --config Release
```

### g++ / clang++ を直接使う

```bash
g++ -O3 -ffast-math -fopenmp -std=c++17 -o nanogpt main.cpp
g++ -O3 -ffast-math -fopenmp -std=c++17 -o gpt2   gpt2.cpp
g++ -O3 -ffast-math -DGPT_USE_DOUBLE -std=c++17 -o nanogpt_gradcheck test_grad.cpp
```

（OpenMP は任意。`-fopenmp` / `/openmp` を外してもビルド・実行できます。）

---

## A. 文字レベル GPT を学習する（完全自己完結）

tiny-shakespeare データセット（約 1MB のテキストファイル）を用意して学習します:

```bash
# input.txt（同梱済み。または再ダウンロード）:
#   curl -sSL -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

nanogpt train input.txt --steps 5000 --layers 4 --embd 128 --heads 4 --block 64 --out ckpt.bin
```

学習中に train/val 損失を表示し、`ckpt.bin` を書き出します。続いてサンプリング:

```bash
nanogpt sample ckpt.bin --tokens 500 --temp 0.8 --topk 40 --prompt "ROMEO:"
```

オプション — `train`: `--steps --lr --batch --block --layers --embd --heads --out
--eval-every --seed --init --ckpt --warmup --min-lr --decay-iters --no-lr-decay
--grad-clip --grad-accum`、`sample`: `--tokens --temp --topk --prompt --seed`。

デフォルトでは学習率は **線形ウォームアップ + コサイン減衰**（nanoGPT の `train.py` と同じ
`get_lr` スケジュール）に従います。`--warmup` ステップ（デフォルト `steps/10`）かけて
ウォームアップし、その後 `--lr` から `--min-lr`（デフォルト `lr/10`）まで `--decay-iters`
（デフォルト: 最終ステップ）で減衰します。学習率を一定にしたい場合は `--no-lr-decay` を
渡します。勾配はグローバル L2 ノルムが `--grad-clip`（デフォルト `1.0`、`0` で無効）に
なるようにクリップされます。`resume` 時はスケジュールが復元されたグローバルステップから
継続するため、ウォームアップは繰り返されません。

`--grad-accum N` は `N` 個のマイクロバッチにわたって勾配を累積してから 1 回の最適化ステップ
を行います（**実効バッチ `--batch × N`**）。nanoGPT の `gradient_accumulation_steps` に
相当し、大きなバッチのメモリを使わずに大バッチを模倣します。

短い学習でも明確に学習しているのが分かります（実測、3 層 / embd 96、CPU のみ）:

```
step     0 | train loss 4.2012 | val loss 4.2000
step   300 | train loss 2.3784 | val loss 2.3962
```

ステップ数やモデルを増やせば損失はさらに下がります（nanoGPT では約 1.5 に到達）。

### 学習の再開（resume）とファインチューニング（finetune）

`train` は、スクラッチから始める代わりに保存済みチェックポイントから続きを学習できます。
`--init` と `--ckpt` を使います:

```bash
# resume: 中断した学習を再開する。重み・AdamW のモーメンタム・ステップ数を復元し、
# 中断した地点から正確に続きを学習する。
nanogpt train input.txt --init resume --ckpt ckpt.bin --steps 2000 --out ckpt.bin

# finetune: 学習済みの重みは保持しつつ、オプティマイザとステップ数はリセットし、
# （通常は）別のテキストファイルを小さめの学習率で学習する。
nanogpt train other.txt --init finetune --ckpt ckpt.bin --steps 500 --lr 3e-4 --out ft.bin
```

- `--init scratch`（デフォルト）— 従来どおりランダム初期化。
- `--init resume` — パラメータ **に加えて** AdamW の状態（`m`・`v`・ステップ数）も
  読み込んで続行。
- `--init finetune` — パラメータのみ読み込む。オプティマイザとステップ数は 0 にリセット。

チェックポイント形式は **v2**（パラメータ + AdamW モーメント + ステップ数）になりました。
旧 **v1**（パラメータのみ）も引き続き読めます。finetune はチェックポイントに焼き込まれた
文字語彙をそのまま再利用します。新しいテキストに元の語彙になかった文字が含まれると、
encode 時にスキップされるため、同じドメイン（例: 別の英語テキスト）で finetune するのが
最良です。

---

## B. 本物の GPT-2（124M … 1.5B）で推論する

事前学習済み重みを一度だけエクスポートします。推奨エクスポータは **numpy + requests のみ**
（torch 不要）で動作します。HuggingFace から GPT-2 の `model.safetensors` をダウンロードし、
Conv1D の重みを nn.Linear のレイアウトに転置して C++ 形式で書き出します。

```bash
pip install numpy requests
python export_gpt2_safetensors.py gpt2 model_gpt2.bin   # 約548MB DL -> 498MB の model_gpt2.bin
```

（別法として `export_gpt2.py gpt2 model_gpt2.bin` は nanoGPT 自身の `model.py` を使います。
こちらは `torch` + `transformers` が必要です。）

BPE ファイル `encoder.json` と `vocab.bpe` も必要です（同梱済み。または `export_gpt2.py`
がダウンロード）。あとは生成するだけ — 純粋な C++、Python 不要:

```bash
gpt2 model_gpt2.bin encoder.json vocab.bpe --prompt "The meaning of life is" --tokens 100 --temp 0.7 --topk 40
```

出力例（GPT-2 124M、実測）:

```
The meaning of life is not the same as the meaning of life. It is not something
a mere human being, a human being, needs. A human being needs to live in the
way we want to live, to live ...
```

`gpt2-medium`、`gpt2-large`、`gpt2-xl` も動きます（エクスポータに名前を渡します）。
CPU では大きなモデルは遅い（素朴な行列積）です。目的は速度ではなく移植性です。
生成は文脈ウィンドウ内で **KV キャッシュ**を使い、毎ステップ全文脈を再計算する場合より
実測で約 **5.5 倍高速**（GPT-2 124M・4 スレッド CPU、短い生成で ~4.6 → ~26 tok/s。系列が
長いほど差は拡大）。

### 自分のテキストで GPT-2 をファインチューニングする

`gpt2 finetune` は、本物の GPT-2 重みをプレーンテキストファイル（その場で BPE トークン化）で
追加学習し、`gpt2` で生成できる新しい `.bin` を書き出します。文字レベル学習器と同じ
forward+backward・AdamW・コサイン LR スケジュール・勾配クリップを共有します。

```bash
# 124M を何かのテキストで fine-tune し、その結果から生成する
gpt2 finetune model_gpt2.bin mytext.txt encoder.json vocab.bpe \
    --steps 200 --block 256 --batch 1 --lr 3e-5 --out model_ft.bin
gpt2 model_ft.bin encoder.json vocab.bpe --prompt "Once upon a time"
```

オプション: `--steps --lr --batch --block --out --eval-every --warmup --min-lr
--decay-iters --no-lr-decay --grad-clip --init finetune|resume --seed`。保存される
`.bin` には AdamW の状態も含まれるため、`--init resume` で中断した地点から fine-tune を
再開できます（通常の `gpt2` 生成は末尾のその状態を無視します）。

速度（このマシン、GPT-2 124M、4 スレッド CPU、`batch 1`）: おおよそ **`--block 128` で
6 秒/step**、**`--block 256` で約 14 秒/step**。したがって 200 ステップ程度の fine-tune は
だいたい **20 分**（`block 128`）〜 **45 分**（`block 256`）、1000 ステップなら数時間です。
CPU のみで未最適化（素朴な行列積、勾配累積なし）で、高速な学習器ではなく動作するデモが
目的です。より大きなモデル（`gpt2-medium`/`large`/`xl`）は比例して遅くなります。

---

## C. Python 版 nanoGPT で学習したモデルを読み込む（重み互換）

nanoGPT 自身の `train.py` が出力したチェックポイント（例: shakespeare_char モデル）を
読み込んで、この C++ 移植版で実行できます。nanoGPT は `nn.Linear` の重みを `[out, in]`
で保存し（Conv1D 転置は不要）、この移植版も nanoGPT の `nn.GELU()` と同じ厳密 erf GELU を
使うため、C++ のロジットは浮動小数点の丸め誤差の範囲で PyTorch と一致します。

```bash
# Python nanoGPT で学習後（out/ckpt.pt と data/.../meta.pkl が出力される）:
python export_nanogpt.py out/ckpt.pt data/shakespeare_char/meta.pkl model.bin
nanogpt sample model.bin --prompt "ROMEO:" --tokens 500 --temp 0.8 --topk 40
```

`verify_nanogpt_compat.py` で **数値的に検証済み** です。これはリポジトリの `model.py` で
nanoGPT モデルを構築し、エクスポートし、同じ入力でロジットを比較します:

```
PyTorch vs C++ logits (16 positions x 65 vocab):
  max abs diff  = 2.384e-07
  argmax agreement = 100.0%
  -> PASS (weight-compatible)
```

したがって greedy 生成は完全に一致し、サンプリング生成も機能的に同一です（RNG のみ異なる）。
なお PyTorch とのビット完全一致は目指していません（カーネルや総和順序が異なる）。約 1e-7 の
一致が実用上の限界です。最も厳密に一致させたい場合は `-ffast-math` なしでビルドしてください。

---

## 検証（すべてこのマシンで実施）

| チェック | 方法 | 結果 |
|---|---|---|
| backprop の正しさ | `nanogpt_gradcheck`（解析 vs 数値、倍精度） | **PASS**、全パラメータテンソルで最大相対誤差 7.5e-6 |
| BPE トークナイザ | `bpe_test` vs 既知の GPT-2 トークン ID | **PASS**（ID 完全一致 + 完全な往復変換） |
| エンドツーエンド学習 | `nanogpt train`（tiny-shakespeare） | loss 4.20 → 2.38、英語テキストを学習 |
| 本物の GPT-2 124M 推論 | `gpt2`（エクスポートした OpenAI 重み） | **動作** — 一貫した英語を生成（上の例） |
| Python nanoGPT との重み互換 | `verify_nanogpt_compat.py`（vs PyTorch ロジット） | **PASS**、最大絶対差 2.4e-7、argmax 100% |

一貫した GPT-2 出力が得られることは、124M の重みが正しいレイアウトで読み込まれ forward が
正しいことの強い証拠です（レイアウトが間違っていれば出力は無意味な文字列になります）。

## メモ / 注意点

- 計算型はデフォルトで `float`。`-DGPT_USE_DOUBLE` で double に切り替わります
  （勾配チェックのターゲットで、厳密なチェックのために使用。有限差分による勾配チェックは
  float では信頼できません）。
- 数値的な勾配チェックは double ビルドでのみ意味があります。そのため
  `nanogpt` のモードではなく別ターゲット（`nanogpt_gradcheck`）にしています。
- BPE の忠実度: GPT-2 の前トークン化は Unicode プロパティの正規表現（`\p{L}`、`\p{N}`）を
  使いますが、依存なしの C++ で再現するのは非現実的です。ここのスプリッタは ASCII 忠実で、
  ASCII 英語では GPT-2 と完全一致（検証済み）し、非 ASCII 文字でのみ乖離します。
  バイトレベル BPE 自体は厳密です。
- GELU: `gpt.h` はデフォルトで厳密な `erf` GELU（nanoGPT の `nn.GELU()`）。GPT-2 エクスポートは
  `tanh` 近似（`gelu_new`、GPT-2 が学習に使ったもの）を設定します。
- dropout は未移植です。nanoGPT の `model.py` には dropout があり（事前学習の既定は `0.0`、
  `shakespeare_char` 設定は `0.2`）ますが、この移植版は常に dropout `0` で動きます。それ以外の
  学習仕様はすべて一致します。ここの toy スケールでは差は小さく、融合 attention と活性化バッファ
  を単純に保つため省略しています。
- KV キャッシュ: 生成は層ごとに key/value をキャッシュし（`GPT::forward_one`）、新しい
  トークンごとに全文脈を再計算する代わりに O(1) 文脈のステップで済みます（フル
  `forward()` とビット単位一致を検証済み）。文脈ウィンドウ内で有効で、生成が
  `block_size` を超えると再計算＋スライドの経路にフォールバックします（学習型の絶対位置
  埋め込みはスライドできないため）。

`input.txt`、`encoder.json`、`vocab.bpe` は便宜のため同梱しています。`.gitignore` して
上記コマンドでユーザーに取得させる運用でも構いません。
