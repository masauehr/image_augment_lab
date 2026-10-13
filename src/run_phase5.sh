#!/bin/bash
# Phase5: 発展拡張（TrivialAugment / RandAugment / Mixup / CutMix）と転移学習との併用
# 使い方: bash src/run_phase5.sh
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# (1) ゼロから学習(SmallCNN): 切り抜き+反転を基準に、自動拡張・Mixup・CutMixを比較（150エポック）
for s in 0 1 2; do
    $PY src/train.py --aug cropflip --seed $s --epochs 150
    $PY src/train.py --aug trivial --seed $s --epochs 150
    $PY src/train.py --aug randaug --seed $s --epochs 150
    $PY src/train.py --aug cropflip --mix mixup --seed $s --epochs 150
    $PY src/train.py --aug cropflip --mix cutmix --seed $s --epochs 150
done

# (2) 転移学習(ImageNet事前学習ResNet18, 全層微調整): 拡張の併用効果（15エポック, lr=1e-4）
for s in 0 1 2; do
    for aug in none cropflip trivial; do
        $PY src/train.py --model resnet18 --lr 1e-4 --aug $aug --seed $s --epochs 15
    done
    $PY src/train.py --model resnet18 --lr 1e-4 --aug cropflip --mix cutmix --seed $s --epochs 15
done

# (3) 転移学習 + 極少量(クラスあたり10枚): 拡張なし vs 切り抜き+反転
for s in 0 1 2; do
    for aug in none cropflip; do
        $PY src/train.py --model resnet18 --lr 1e-4 --aug $aug --seed $s --epochs 15 --n_per_class 10
    done
done
echo "PHASE5_DONE"
