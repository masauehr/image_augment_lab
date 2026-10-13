"""Phase2〜: 小さなCNNで hymenoptera を分類し、拡張あり/なしを比較する。

使い方:
    .venv/bin/python src/train.py --aug none  --seed 0
    .venv/bin/python src/train.py --aug basic --seed 0

出力:
    outputs/{aug}_seed{seed}_history.csv  エポックごとのloss/精度
    outputs/{aug}_seed{seed}_curve.png    学習曲線
    outputs/results.csv                   実行ごとの最終結果（追記）
"""
import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import v2

plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "DejaVu Sans"]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "hymenoptera"
OUT = ROOT / "outputs"
SIZE = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet統計

# 拡張の定義。評価用(val/test)には拡張を入れず、リサイズ+正規化のみ
AUG_SETS = {
    "none": [v2.Resize((SIZE, SIZE))],
    "flip": [v2.Resize((SIZE, SIZE)), v2.RandomHorizontalFlip()],
    "crop": [v2.RandomResizedCrop(SIZE, scale=(0.6, 1.0))],
    "color": [v2.Resize((SIZE, SIZE)), v2.ColorJitter(0.3, 0.3, 0.3, 0.05)],
    "cropflip": [v2.RandomResizedCrop(SIZE, scale=(0.6, 1.0)), v2.RandomHorizontalFlip()],
    "trivial": [v2.Resize((SIZE, SIZE)), v2.TrivialAugmentWide()],
    "randaug": [v2.Resize((SIZE, SIZE)), v2.RandAugment(num_ops=2, magnitude=9)],
    "basic": [
        v2.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(0.3, 0.3, 0.3, 0.05),
    ],
}


def make_transform(aug: str | None) -> v2.Compose:
    """aug=Noneなら評価用。それ以外は学習用拡張+共通の変換"""
    head = [v2.Resize((SIZE, SIZE))] if aug is None else AUG_SETS[aug]
    return v2.Compose(
        [v2.ToImage(), *head, v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
    )


class SmallCNN(nn.Module):
    """畳み込み3層の小さなCNN（意図的に小さく、過学習の様子を見やすくする）"""

    def __init__(self, n_classes: int = 2):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def make_model(name: str, n_classes: int) -> nn.Module:
    """small=自作CNN（ゼロから学習） / resnet18=ImageNet事前学習済み（転移学習）"""
    if name == "small":
        return SmallCNN(n_classes)
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def subsample(ds, n_per_class: int, seed: int) -> Subset:
    """クラスごとにn枚をシード付きで無作為抽出する（学習データ枚数の影響を見る用）"""
    rng = random.Random(seed)
    idx = []
    for c in range(len(ds.classes)):
        members = [i for i, (_, y) in enumerate(ds.samples) if y == c]
        idx += rng.sample(members, min(n_per_class, len(members)))
    return Subset(ds, idx)


def run_epoch(model, loader, device, loss_fn, optimizer=None, mixer=None):
    """optimizerありなら学習、なしなら評価。(loss, acc)を返す"""
    model.train(optimizer is not None)
    total_loss = correct = n = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hard = y  # 精度計算用（Mixup/CutMix後はyがソフトラベルになるため元のラベルを保持）
            if mixer is not None and optimizer is not None:
                x, y = mixer(x, y)
            out = model(x)
            loss = loss_fn(out, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y_hard).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", choices=AUG_SETS, default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--model", choices=["small", "resnet18"], default="small")
    ap.add_argument("--mix", choices=["none", "mixup", "cutmix"], default="none")
    ap.add_argument("--n_per_class", type=int, default=0, help="学習データのクラスあたり枚数（0=全部）")
    a = ap.parse_args()

    set_seed(a.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_ds = datasets.ImageFolder(DATA / "train", make_transform(a.aug))
    if a.n_per_class:
        train_ds_full = train_ds
        train_ds = subsample(train_ds, a.n_per_class, a.seed)
    else:
        train_ds_full = train_ds
    val_ds = datasets.ImageFolder(DATA / "val", make_transform(None))
    test_ds = datasets.ImageFolder(DATA / "test", make_transform(None))
    g = torch.Generator().manual_seed(a.seed)
    train_dl = DataLoader(train_ds, a.batch, shuffle=True, generator=g)
    val_dl = DataLoader(val_ds, a.batch)
    test_dl = DataLoader(test_ds, a.batch)

    n_cls = len(train_ds_full.classes)
    model = make_model(a.model, n_cls).to(device)
    mixer = {"none": None, "mixup": v2.MixUp(num_classes=n_cls), "cutmix": v2.CutMix(num_classes=n_cls)}[a.mix]
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()

    hist = []
    for ep in range(1, a.epochs + 1):
        tl, ta = run_epoch(model, train_dl, device, loss_fn, opt, mixer)
        vl, va = run_epoch(model, val_dl, device, loss_fn)
        hist.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
        print(f"ep{ep:3d} train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}")
    # testは最終エポックのモデルで1回だけ評価（test結果でエポックを選ばない）
    _, test_acc = run_epoch(model, test_dl, device, loss_fn)

    OUT.mkdir(exist_ok=True)
    label = a.aug + ("" if a.mix == "none" else f"+{a.mix}")
    tag = ("" if a.model == "small" else f"{a.model}_") + f"{label}_seed{a.seed}" + (f"_n{a.n_per_class}" if a.n_per_class else "")
    with open(OUT / f"{tag}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hist[0].keys())
        w.writeheader()
        w.writerows(hist)

    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="val")
    ax[0].set(title="loss", xlabel="epoch")
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="val")
    ax[1].set(title="精度", xlabel="epoch", ylim=(0.4, 1.02))
    for x in ax:
        x.legend()
        x.grid(alpha=0.3)
    fig.suptitle(f"{a.model} aug={label} seed={a.seed} n/class={a.n_per_class or '全部'}")
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_curve.png", dpi=100)

    last = hist[-1]
    res = OUT / "results.csv"
    new = not res.exists()
    with open(res, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "aug", "seed", "epochs", "n_per_class", "train_acc", "val_acc", "test_acc", "gap(train-val)"])
        w.writerow([a.model, label, a.seed, a.epochs, a.n_per_class or "all", f"{last['train_acc']:.3f}", f"{last['val_acc']:.3f}",
                    f"{test_acc:.3f}", f"{last['train_acc'] - last['val_acc']:.3f}"])
    print(f"最終: train {last['train_acc']:.3f} val {last['val_acc']:.3f} test {test_acc:.3f}")


if __name__ == "__main__":
    main()
