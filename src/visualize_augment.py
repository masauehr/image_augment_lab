"""Phase1: 1枚の画像に各拡張を適用して並べて表示する。

使い方: .venv/bin/python src/visualize_augment.py
出力:   outputs/augment_gallery.png
"""
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 日本語フォント（macOS標準）。見つからなければ既定フォントのまま
plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "DejaVu Sans"]
import torch
from PIL import Image
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
SEED = 0
SIZE = 224

# 名前 → 拡張（1つずつ単独で見比べる）
AUGMENTS = {
    "水平反転 (p=1)": v2.RandomHorizontalFlip(p=1.0),
    "垂直反転 (p=1)": v2.RandomVerticalFlip(p=1.0),
    "回転 ±30°": v2.RandomRotation(degrees=30),
    "切り抜き+拡大 (scale .4-1)": v2.RandomResizedCrop(SIZE, scale=(0.4, 1.0)),
    "アフィン (移動/拡大縮小)": v2.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    "色調 (明るさ/彩度/色相)": v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
    "グレースケール (p=1)": v2.RandomGrayscale(p=1.0),
    "ぼかし": v2.GaussianBlur(kernel_size=9, sigma=(2.0, 4.0)),
    "透視変換 (p=1)": v2.RandomPerspective(distortion_scale=0.5, p=1.0),
    "TrivialAugment": v2.TrivialAugmentWide(),
}


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    path = sorted((ROOT / "data/processed/hymenoptera/train/ants").glob("*.jpg"))[0]
    img = v2.Resize((SIZE, SIZE))(Image.open(path).convert("RGB"))

    n_cols = 4
    items = [("元画像", img)]
    items += [(name, aug(img)) for name, aug in AUGMENTS.items()]
    n_rows = -(-len(items) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, im) in zip(axes.flat, items):
        ax.imshow(im)
        ax.set_title(name, fontsize=9)
    out = ROOT / "outputs" / "augment_gallery.png"
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
