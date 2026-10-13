"""Notebook共通のユーティリティ（学習用の小さな道具箱）。

各Notebookの先頭で `from nb_utils import *` して使う。
- FAST: 環境変数 NB_FAST=1 のとき、エポック数を大幅に減らす（動作確認用）。通常は使わない。
"""
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

import train as T  # 同じsrc/内のtrain.py（日本語フォント設定もここで行われる）
from train import DATA, OUT, ROOT, fit, plot_history, summarize  # noqa: F401

FAST = os.environ.get("NB_FAST") == "1"


def ep(n: int) -> int:
    """エポック数。FASTモードでは1/25に縮める（動作確認用）"""
    return max(2, n // 25) if FAST else n


def show_images(images, titles=None, cols=4, size=3.0):
    """画像（PIL/Tensor）を格子状に並べて表示する"""
    rows = -(-len(images) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, (size + 0.3) * rows))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for i, (ax, im) in enumerate(zip(np.atleast_1d(axes).flat, images)):
        if torch.is_tensor(im):
            im = im.detach().cpu().float()
            if im.min() < 0:  # 正規化済みなら元に戻す
                m = torch.tensor(T.MEAN).view(3, 1, 1)
                s = torch.tensor(T.STD).view(3, 1, 1)
                im = im * s + m
            im = im.clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(im)
        if titles is not None:
            ax.set_title(titles[i], fontsize=9)
    fig.tight_layout()
    return fig


def run_seeds(seeds, **kw):
    """同じ条件を複数シードで学習する。(結果のDataFrame, 各実行のdictのリスト)を返す"""
    rows, results = [], []
    t0 = time.time()
    for s in seeds:
        r = fit(seed=s, verbose=False, **kw)
        rows.append(summarize(r))
        results.append(r)
        print(f"  seed={s}: train {rows[-1]['train_acc']:.3f}  val {rows[-1]['val_acc']:.3f}  "
              f"test {rows[-1]['test_acc']:.3f}  ({time.time() - t0:.0f}秒経過)")
    return pd.DataFrame(rows), results


def mean_table(df, by=("aug",)):
    """シード平均±標準偏差の比較表を作る"""
    g = df.groupby(list(by), sort=False)[["train_acc", "val_acc", "test_acc", "gap"]]
    mean, std = g.mean(), g.std().fillna(0)
    out = mean.round(3).astype(str) + " ±" + std.round(3).astype(str)
    out["n_seeds"] = g.size()
    return out
