# image-augment-lab

画像分類の深層学習で、データ拡張（反転・切り抜き・色調変更など）により少量の教師データを
水増しするノウハウを学ぶサンプルプロジェクト。実体は [image_augment_lab](.) プロジェクト。

## 概要

- 目的は精度の追求ではなく、データ拡張の効果・判断基準・落とし穴の習得。
- データは少量（2〜3クラス × 各50〜100枚程度）。小さなCNNをCPU/MPSで数分以内に学習する。
- PyTorch + torchvision（`transforms.v2`）を使用。乱数シード固定で「拡張の有無だけ」を変えて比較する。

## フェーズ計画

| Phase | 内容 |
|---|---|
| 0 | 環境構築・少量データセット準備・train/val/test分割（拡張前に分割） |
| 1 | 拡張操作の可視化（1枚から何が作れるか） |
| 2 | ベースライン（拡張なし）で過学習を確認 |
| 3 | 拡張あり vs なしの精度比較 |
| 4 | 拡張の種類ごとのアブレーション、データ枚数を変えた効果検証 |
| 5 | RandAugment / Mixup / CutMix、転移学習との併用、落とし穴の整理（完了） |

## Notebookで学ぶ

全フェーズを初心者向けの解説付きJupyter Notebookにしてある（00に演習全体のゴール・用語集あり。各演習は変更箇所と観察ポイントまで具体的に記載）（`notebooks/00_setup_data` 〜 `05_advanced`）。上から順に実行しながら読み進める。

| Notebook | フェーズ | 目安時間 |
|---|---|---|
| 00_setup_data | Phase0 データ取得・分割・リーク確認 | 1〜2分 |
| 01_visualize_augment | Phase1 拡張の可視化・強さの違い | 1分 |
| 02_baseline | Phase2 ベースライン・過学習 | 1〜2分 |
| 03_augment_vs_none | Phase3 拡張あり/なし比較 | 約3分 |
| 04_ablation | Phase4 アブレーション・学習枚数 | 約15分 |
| 05_advanced | Phase5 自動拡張・Mixup/CutMix・転移学習 | 約15分 |

- Notebookを実行しなくても、`notebooks/README.md` に各Notebookの結果（図・表・読み取れること）をまとめてある（予習・答え合わせ用）。
- 起動: `.venv/bin/jupyter lab notebooks/`
- `QUICK = True`（1シード・短時間）/ `False`（3シード・MANUAL.mdの記載値と同条件）。`NB_FAST=1` で動作確認のみ可能。
- `No module named 'torch'` が出たらカーネル選択で「Python (image_augment_lab)」（`.venv`）を選ぶ。無ければ `.venv/bin/python -m ipykernel install --user --name image_augment_lab --display-name "Python (image_augment_lab)"` で登録する。
- Notebookは `outputs/results.csv` を書き換えない（CLI実行とは独立）。出力を消した状態でコミットする。

## 主な拡張操作

反転（水平/垂直）、回転、ランダム切り抜き+拡大（RandomResizedCrop）、アフィン変換、
ColorJitter、グレースケール、ぼかし、ノイズ、Random Erasing、Mixup、CutMix、自動拡張（RandAugment / TrivialAugment）。
各操作の使いどころ・注意点は [MANUAL.md](MANUAL.md) を参照。

## 重要な注意点

- 拡張は**学習データのみ**に適用（val/testはリサイズ・正規化のみ）。
- データ分割は拡張の**前**（同一画像由来の派生画像が分割をまたぐとデータリークになる）。
- ラベルを壊す拡張（数字の左右反転、色が手がかりのクラスでの色相変更など）は避ける。

## 現状

2026-10-13: Phase0（環境構築・データ取得・分割）・Phase1（拡張の可視化）・Phase2（拡張なしベースライン）・Phase3（拡張あり比較）・Phase4（アブレーション・枚数の影響）・Phase5（発展手法・転移学習）完了。計画した全フェーズが終了。

- データ: 自前写真は使わず公開データのみ。主データは PyTorch 公式チュートリアルの hymenoptera（アリ vs ハチ、train 243 / val 76 / test 77枚）、追試に Flowers102 の2〜3クラス。
- スクリプト: `src/prepare_data.py`（取得・分割、シード42固定）、`src/visualize_augment.py`（拡張ギャラリー出力）。
- `src/train.py`: SmallCNN（畳み込み3層）で学習。`--aug none|flip|crop|color|basic --seed N --epochs N`。結果は `outputs/results.csv` に追記。
- ベースライン（拡張なし, 3シード平均）: train 0.859 / val 0.719 / test 0.809。train-val差は約0.14で、val lossが下がらず過学習を確認。valが約76枚と少なくシード間ばらつきが大きいため、比較は複数シードで行う。
- 拡張あり(basic: 切り抜き+反転+色調) vs なし（3シード平均、最終エポック）:
  - 60エポック: val 0.715 vs 0.719、test 0.757 vs 0.809（拡張ありは学習が遅く差が出ない）
  - 150エポック: val 0.732 vs 0.711、test 0.796 vs 0.701、train-val差 0.089 vs 0.235（拡張の効果が現れる）
  - 教訓: 拡張は過学習を抑える代わりに収束が遅い。十分な長さ学習してから比較する。
- アブレーション（150エポック・3シード、test平均）: 拡張なし 0.701 / 水平反転 0.757 / **ランダム切り抜き 0.835** / 色調のみ 0.701（効果なし）/ basic全部入り 0.796。効く拡張を見極めて絞る価値がある。
- 学習枚数の影響（クラスあたり10/30/60枚、拡張なし vs basic）: train-val差はどの枚数でも縮むが、testが改善したのは60枚（0.658→0.731）のみ。10・30枚では改善なし。拡張はデータの代わりにならない。
- 発展手法（ゼロから学習150エポック、test平均）: RandAugment 0.844 / 切り抜き 0.835 / TrivialAugment 0.792 / 拡張なし 0.701。Mixup（0.753）・CutMix（0.714）は小さなCNNでは逆効果（未学習気味）。
- 転移学習（ImageNet事前学習ResNet18）: 拡張なしでtest 0.892（ゼロから学習の0.701から+19pt）、切り抜き+反転で0.939。クラス10枚でも0.78〜0.81（ゼロから学習は0.64）。少量データでは転移学習が最大の改善要因で、拡張は上積み。
- 指針: ①まず転移学習 ②拡張は切り抜き+反転から ③迷えばRandAugment ④1種類ずつ試す ⑤十分なエポックで比較 ⑥複数シード平均で比べる（詳細は MANUAL.md「実験から得た実践的な指針」）。
- 一括実行: `bash src/run_phase4.sh`（アブレーション・枚数）/ `bash src/run_phase5.sh`（発展手法・転移学習）
- `src/nb_utils.py`: Notebook共通の道具箱（画像表示・複数シード実行・比較表）。`src/train.py` に `fit()` / `plot_history()` / `summarize()` を切り出し、CLIとNotebookで共用。
- 実行環境: Python 3.14 + torch 2.14.1 / torchvision 0.29.1、Apple Silicon の MPS 利用可。

## 次のステップ（検討中）

詳細は PLAN.md の「次のステップ」を参照。推奨順は **A 評価を固める**（交差検証・混同行列・Grad-CAM）→ **B 予測・検証システム化**（モデル保存・推論スクリプト・確信度・検証レポート）→ **C 気象への応用**（台風・天気図）。
気象では**左右反転が物理的に誤り**（台風の回転方向が逆になる）など、「ラベルを壊す拡張は使えない」の実践になる。B・Cは別プロジェクトとして作る案が有力。
