[English](README.md) | [日本語](README.ja.md)

# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

> **C++ 移植版について** — このリポジトリには依存ライブラリなしの C++17 移植版
> ([`cpp/`](cpp/)) が含まれます。詳しくは [`cpp/README.ja.md`](cpp/README.ja.md) を参照してください。

---

**2025年11月 更新** nanoGPT には [nanochat](https://github.com/karpathy/nanochat) という
新しく改良された兄弟版があります。多くの場合、あなたが使いたい/探しているのは nanochat の方でしょう。
nanoGPT（このリポジトリ）はもう非常に古く非推奨ですが、後世のために残しておきます。

---

中規模の GPT を学習/ファインチューニングするための、最もシンプルで最速のリポジトリです。
[minGPT](https://github.com/karpathy/minGPT) を「教育よりも実用」を優先して書き直したものです。
現在も活発に開発中ですが、いまのところ `train.py` は OpenWebText 上で GPT-2 (124M) を再現でき、
単一の 8×A100 40GB ノードで約 4 日の学習で動きます。コード自体は素朴で読みやすく、
`train.py` は約 300 行の定型的な学習ループ、`model.py` は約 300 行の GPT モデル定義で、
オプションで OpenAI の GPT-2 重みを読み込めます。それだけです。

![repro124m](assets/gpt2_124M_loss.png)

コードがとてもシンプルなので、自分のニーズに合わせて改造したり、新しいモデルをスクラッチから
学習したり、事前学習済みチェックポイントをファインチューニングしたりするのが容易です
（例: 現在利用できる最大の出発点は OpenAI の GPT-2 1.3B モデル）。

## インストール

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

依存関係:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `transformers` — huggingface transformers 用（GPT-2 チェックポイントの読み込み） <3
- `datasets` — huggingface datasets 用（OpenWebText をダウンロード+前処理したい場合） <3
- `tiktoken` — OpenAI の高速 BPE <3
- `wandb` — オプションのロギング <3
- `tqdm` — プログレスバー <3

## クイックスタート

深層学習の専門家ではなく、とにかく魔法を体感して雰囲気を掴みたいだけなら、最速の始め方は
シェイクスピア作品で文字レベル GPT を学習することです。まず 1MB の単一ファイルとして
ダウンロードし、生テキストから整数の大きな列へ変換します:

```sh
python data/shakespeare_char/prepare.py
```

これでそのデータディレクトリに `train.bin` と `val.bin` が作られます。次は GPT を学習します。
サイズはシステムの計算リソースに大きく依存します。

**GPU がある場合**。素晴らしい。[config/train_shakespeare_char.py](config/train_shakespeare_char.py)
の設定で小さな GPT をすぐ学習できます:

```sh
python train.py config/train_shakespeare_char.py
```

中を覗くと、文脈サイズ最大 256 文字、384 特徴チャンネル、各層 6 ヘッドの 6 層 Transformer を
学習しているのが分かります。1 枚の A100 GPU でこの学習は約 3 分、最良の検証損失は 1.4697 です。
設定に従い、モデルのチェックポイントは `--out_dir` の `out-shakespeare-char` に書き出されます。
学習が終わったら、サンプリングスクリプトをこのディレクトリに向けて最良モデルからサンプリングできます:

```sh
python sample.py --out_dir=out-shakespeare-char
```

いくつかのサンプルが生成されます。例:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol `¯\_(ツ)_/¯`。GPU で 3 分学習した文字レベルモデルにしては悪くありません。代わりに事前学習済み
GPT-2 をこのデータセットでファインチューニングすれば、もっと良い結果が得られる可能性が高いです
（後述のファインチューニングの節を参照）。

**macbook（または安価なマシン）しかない場合**。心配ありません。それでも GPT を学習できますが、
少し設定を控えめにします。最新の PyTorch nightly を入れることを推奨します
（[ここ](https://pytorch.org/get-started/locally/) でインストール時に選択）。ただ、それがなくても
シンプルな学習は次のようにできます:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

ここでは GPU ではなく CPU で動かすので、`--device=cpu` を設定し、PyTorch 2.0 の compile も
`--compile=False` で無効化します。評価はややノイジーだが高速な推定になり（`--eval_iters=20`、
200 から削減）、文脈サイズは 256 ではなく 64 文字、バッチサイズも 64 ではなく 12 です。
はるかに小さい Transformer（4 層、4 ヘッド、埋め込みサイズ 128）を使い、反復回数を 2000 に減らします
（それに合わせて通常 `--lr_decay_iters` で学習率を max_iters 付近まで減衰させます）。ネットワークが
小さいので正則化も緩めます（`--dropout=0.0`）。これでも約 3 分で動きますが、損失は 1.88 程度、
サンプルも劣りますが、それでも楽しめます:

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

こんなサンプルが生成されます:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

CPU で約 3 分にしては悪くなく、正しい文字の雰囲気の片鱗が見えます。もっと待てるなら、
ハイパーパラメータの調整、ネットワークサイズや文脈長（`--block_size`）の拡大、学習時間の延長などを
自由に試してください。

最後に、Apple Silicon の Macbook で最近の PyTorch を使う場合は `--device=mps`
（"Metal Performance Shaders" の略）を付けてください。PyTorch がオンチップ GPU を使い、学習を
*大幅に* 高速化（2〜3 倍）でき、より大きなネットワークが使えます。詳しくは
[Issue 28](https://github.com/karpathy/nanoGPT/issues/28) を参照。

## GPT-2 の再現

より本格的な深層学習の専門家は、GPT-2 の結果を再現したいかもしれません。ではやってみましょう。
まずデータセットをトークン化します。ここでは OpenAI の（非公開）WebText のオープンな再現版である
[OpenWebText](https://openwebtext2.readthedocs.io/en/latest/) を使います:

```sh
python data/openwebtext/prepare.py
```

これで [OpenWebText](https://huggingface.co/datasets/openwebtext) データセットをダウンロード・
トークン化します。GPT2 BPE トークン ID を 1 本の列として持つ `train.bin` と `val.bin` が
生の uint16 バイトで作られます。次に学習を開始します。GPT-2 (124M) を再現するには、少なくとも
8×A100 40GB ノードが必要で、次を実行します:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

PyTorch Distributed Data Parallel (DDP) を使って約 4 日動き、損失は約 2.85 まで下がります。
GPT-2 モデルを OWT で評価すると検証損失は約 3.11 ですが、ファインチューニングすると（明らかな
ドメインギャップにより）約 2.85 まで下がり、2 つのモデルはほぼ一致します。

クラスタ環境で複数の GPU ノードに恵まれているなら、例えば 2 ノードで次のように GPU をブン回せます:

```sh
# 例として IP 123.456.123.456 の 1 台目（マスター）ノードで実行:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# ワーカーノードで実行:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

インターコネクトをベンチマーク（例: iperf3）しておくと良いです。特に Infiniband がない場合は
上記の起動コマンドの前に `NCCL_IB_DISABLE=1` を付けてください。マルチノード学習は動きますが、
おそらく _這うほど遅く_ なります。デフォルトではチェックポイントが定期的に `--out_dir` に
書き出されます。`python sample.py` だけでモデルからサンプリングできます。

最後に、単一 GPU で学習するには単に `python train.py` を実行します。すべての引数を見てください。
スクリプトは非常に読みやすく、改造しやすく、透明であろうとしています。ニーズに応じて、多くの
変数を調整したくなるはずです。

## ベースライン

OpenAI の GPT-2 チェックポイントを使えば、openwebtext のベースラインを用意できます。数値は
次のように取得します:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

train と val で以下の損失が観測されます:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

ただし、GPT-2 は（非公開・未リリースの）WebText で学習された一方、OpenWebText はその
ベストエフォートなオープン再現に過ぎない点に注意が必要です。つまりデータセットのドメインギャップが
あります。実際、GPT-2 (124M) のチェックポイントを OWT で少しファインチューニングすると損失は
約 2.85 まで下がります。これが再現に関するより適切なベースラインとなります。

## ファインチューニング

ファインチューニングは学習と何ら変わりません。事前学習済みモデルから初期化し、より小さい学習率で
学習するだけです。新しいテキストで GPT をファインチューニングする例は、`data/shakespeare` に行って
`prepare.py` を実行し、tiny shakespeare データセットをダウンロードして、GPT-2 の OpenAI BPE
トークナイザで `train.bin` と `val.bin` にレンダリングしてください。OpenWebText と違い数秒で終わります。
ファインチューニングはごく短時間、例えば単一 GPU で数分で済みます。例:

```sh
python train.py config/finetune_shakespeare.py
```

これは `config/finetune_shakespeare.py` の設定上書きを読み込みます（あまり調整はしていませんが）。
基本的には `init_from` で GPT2 チェックポイントから初期化し、より短く小さい学習率にする以外は通常通り
学習します。メモリ不足になったらモデルサイズを下げる（`{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`）か、
`block_size`（文脈長）を下げてみてください。最良のチェックポイント（最小の検証損失）は `out_dir`
（設定ファイルによりデフォルトで `out-shakespeare`）に置かれます。その後 `sample.py --out_dir=out-shakespeare`
を実行できます:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

おっと GPT、少し暗いところに入り込んでいますね。設定内のハイパーパラメータはあまり調整して
いないので、ぜひ試してみてください！

## サンプリング / 推論

`sample.py` を使って、OpenAI がリリースした事前学習済み GPT-2 モデル、または自分で学習した
モデルからサンプリングできます。例えば、利用可能な最大の `gpt2-xl` からサンプリングする方法:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

自分で学習したモデルからサンプリングしたい場合は、`--out_dir` でコードを適切に向けてください。
ファイルのテキストでモデルにプロンプトを与えることもできます。例:
```python sample.py --start=FILE:prompt.txt```。

## 効率に関するメモ

単純なモデルのベンチマークとプロファイリングには `bench.py` が便利です。`train.py` の学習ループの
核心部分と同一ですが、その他の複雑な部分の多くを省いています。

なお、コードはデフォルトで [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/) を使います。
執筆時点（2022年12月29日）では nightly リリースで `torch.compile()` が使えます。この 1 行の改善は
顕著で、例えば反復時間を約 250ms/iter から 135ms/iter に短縮します。PyTorch チーム、お見事！

## TODO

- DDP の代わりに FSDP を調査・追加する
- 標準的な評価（例: LAMBADA? HELM? など）でゼロショットのパープレキシティを評価する
- ファインチューニングスクリプトを調整する。ハイパーパラメータが良くないと思う
- 学習中の線形なバッチサイズ増加のスケジュール
- 他の埋め込み（rotary、alibi）を組み込む
- チェックポイントで optim バッファをモデルパラメータから分離する（多分）
- ネットワークの健全性まわりの追加ロギング（例: 勾配クリップイベント、大きさ）
- より良い初期化などのさらなる調査

## トラブルシューティング

このリポジトリはデフォルトで PyTorch 2.0（すなわち `torch.compile`）を使う点に注意してください。
これは比較的新しく実験的で、すべてのプラットフォーム（例: Windows）でまだ利用できません。
関連するエラーメッセージが出る場合は `--compile=False` フラグを追加して無効化してみてください。
コードは遅くなりますが、少なくとも動きます。

このリポジトリ、GPT、言語モデリングの背景については、私の
[Zero To Hero シリーズ](https://karpathy.ai/zero-to-hero.html) を見ると役立つかもしれません。
特に [GPT 動画](https://www.youtube.com/watch?v=kCc8FmEb1nY) は、言語モデリングの前提知識が
ある人に人気です。

さらなる質問/議論は Discord の **#nanoGPT** にお気軽にどうぞ:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 謝辞

nanoGPT のすべての実験は、私のお気に入りのクラウド GPU プロバイダである
[Lambda labs](https://lambdalabs.com) の GPU で動いています。nanoGPT のスポンサーになってくれた
Lambda labs に感謝します！
