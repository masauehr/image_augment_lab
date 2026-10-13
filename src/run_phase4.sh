#!/bin/bash
# Phase4: アブレーション（拡張を1種類ずつ）と学習データ枚数の影響を一括実行する
# 使い方: bash src/run_phase4.sh
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# (1) アブレーション: 150エポック・3シード（none/basic/flip(seed0)は実行済み）
for s in 1 2; do $PY src/train.py --aug flip --seed $s --epochs 150; done
for s in 0 1 2; do
    $PY src/train.py --aug crop --seed $s --epochs 150
    $PY src/train.py --aug color --seed $s --epochs 150
done

# (2) 学習データ枚数の影響: クラスあたり10/30/60枚、拡張なし vs basic（300エポック）
for n in 10 30 60; do
    for aug in none basic; do
        for s in 0 1 2; do
            $PY src/train.py --aug $aug --seed $s --epochs 300 --n_per_class $n
        done
    done
done
echo "PHASE4_DONE"
