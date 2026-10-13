# image_augment_lab

画像を教師データとする深層学習（画像分類）で、**データ拡張（Data Augmentation）によって
少ない教師データを水増しするノウハウ**を学ぶためのサンプルプロジェクト。

詳しくは [MANUAL.md](MANUAL.md) / [image-augment-lab.md](image-augment-lab.md)（pc_docs運用マニュアルのコピー）を参照。

## 目的

- 反転・回転・切り抜き・色調変更などの操作で、教師データを実質的に増やす方法を体験的に学ぶ。
- 「拡張なし」と「拡張あり」で精度がどう変わるかを、少量データで比較して実感する。
- どの拡張が効き、どの拡張が逆効果になるか（例: 数字の左右反転）の判断軸を身につける。
- 精度を上げることよりも、**ノウハウの習得と再利用可能なコード・メモの蓄積**が主目的。

## データ

自前写真は使わず、公開データのみ使用する。主データは PyTorch 公式チュートリアルの
hymenoptera（アリ vs ハチ）、追試に Flowers102 の一部。詳細は [PLAN.md](PLAN.md) を参照。

## クイックスタート

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/prepare_data.py       # データ取得・train/val/test分割
.venv/bin/python src/visualize_augment.py  # 拡張ギャラリー → outputs/augment_gallery.png
.venv/bin/python src/train.py --aug none --seed 0 --epochs 60   # 学習（--aug none/flip/crop/color/basic）
```

## 方針

- データ数は少なくてよい（例: 2〜3クラス × 各50〜100枚程度）。少ないほど拡張の効果が見えやすい。
- 小さなCNNをCPU/Apple Silicon（MPS）で数分以内に学習できる規模に抑える。
- 比較実験は「条件を1つだけ変える」ことを徹底する（再現性のため乱数シード固定）。
- フレームワークは PyTorch + torchvision（`transforms.v2`）を使う。

## 進め方（サマリー）

詳細は [PLAN.md](PLAN.md) を参照。

1. 環境構築・小規模データセットの用意
2. 拡張操作の可視化（1枚の画像から何が作れるかを目で確認）
3. ベースライン学習（拡張なし）で過学習を確認
4. 拡張ありで学習し、精度を比較
5. 拡張の種類ごとの効果を切り分け（アブレーション）
6. 発展手法（RandAugment / Mixup / CutMix）と注意点の整理

## ディレクトリ構成

```
image_augment_lab/
├── README.md          # このファイル
├── CLAUDE.md          # Claude Code向けプロジェクト固有ルール
├── MANUAL.md          # データ拡張のノウハウ集（手法・判断基準・落とし穴）
├── PLAN.md            # フェーズ計画・進捗
├── requirements.txt   # 依存パッケージ
├── data/
│   ├── raw/           # 元画像（git管理外）
│   └── processed/     # train/val/test分割済みデータ（git管理外）
├── notebooks/         # 可視化・実験用notebook
├── src/               # prepare_data.py（取得・分割）/ visualize_augment.py（拡張の可視化）/ train.py（学習・比較）
└── outputs/           # 学習曲線・比較表・サンプル画像（git管理外）
```

## 関連プロジェクト

- [lichens](../lichens/) — 地衣類の写真データ。実写真を使う発展課題の候補データ源。
