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


def fit(aug="none", seed=0, epochs=30, lr=1e-3, batch=32, n_per_class=0, model_name="small", mix="none", verbose=True):
    """学習を1回実行し、結果をdictで返す（CLIとNotebookの共通処理）。

    返り値: dict(hist=エポックごとの履歴, test_acc, label, tag, model, aug, seed, epochs, n_per_class, model_obj)
    """
    set_seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_ds_full = datasets.ImageFolder(DATA / "train", make_transform(aug))
    train_ds = subsample(train_ds_full, n_per_class, seed) if n_per_class else train_ds_full
    val_ds = datasets.ImageFolder(DATA / "val", make_transform(None))
    test_ds = datasets.ImageFolder(DATA / "test", make_transform(None))
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch, shuffle=True, generator=g)
    val_dl = DataLoader(val_ds, batch)
    test_dl = DataLoader(test_ds, batch)

    n_cls = len(train_ds_full.classes)
    model = make_model(model_name, n_cls).to(device)
    mixer = {"none": None, "mixup": v2.MixUp(num_classes=n_cls), "cutmix": v2.CutMix(num_classes=n_cls)}[mix]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    hist = []
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(model, train_dl, device, loss_fn, opt, mixer)
        vl, va = run_epoch(model, val_dl, device, loss_fn)
        hist.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
        if verbose:
            print(f"ep{ep:3d} train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}")
    # testは最終エポックのモデルで1回だけ評価（test結果でエポックを選ばない）
    _, test_acc = run_epoch(model, test_dl, device, loss_fn)

    label = aug + ("" if mix == "none" else f"+{mix}")
    tag = ("" if model_name == "small" else f"{model_name}_") + f"{label}_seed{seed}" + (f"_n{n_per_class}" if n_per_class else "")
    return dict(hist=hist, test_acc=test_acc, label=label, tag=tag, model=model_name, aug=aug, seed=seed,
                epochs=epochs, n_per_class=n_per_class, model_obj=model)


def plot_history(r: dict, path: Path | None = None):
    """学習曲線（loss・精度）を描く。pathを渡すと画像として保存する"""
    hist = r["hist"]
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
    fig.suptitle(f"{r['model']} aug={r['label']} seed={r['seed']} n/class={r['n_per_class'] or '全部'}")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=100)
    return fig


def summarize(r: dict) -> dict:
    """1回の実行結果を、比較表の1行（dict）にまとめる"""
    last = r["hist"][-1]
    return dict(model=r["model"], aug=r["label"], seed=r["seed"], epochs=r["epochs"],
                n_per_class=r["n_per_class"] or "all", train_acc=round(last["train_acc"], 3),
                val_acc=round(last["val_acc"], 3), test_acc=round(r["test_acc"], 3),
                gap=round(last["train_acc"] - last["val_acc"], 3))


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

    r = fit(a.aug, a.seed, a.epochs, a.lr, a.batch, a.n_per_class, a.model, a.mix)
    hist, tag = r["hist"], r["tag"]

    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{tag}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hist[0].keys())
        w.writeheader()
        w.writerows(hist)
    plot_history(r, OUT / f"{tag}_curve.png")

    last = hist[-1]
    res = OUT / "results.csv"
    new = not res.exists()
    with open(res, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "aug", "seed", "epochs", "n_per_class", "train_acc", "val_acc", "test_acc", "gap(train-val)"])
        w.writerow([a.model, r["label"], a.seed, a.epochs, a.n_per_class or "all", f"{last['train_acc']:.3f}",
                    f"{last['val_acc']:.3f}", f"{r['test_acc']:.3f}", f"{last['train_acc'] - last['val_acc']:.3f}"])
    print(f"最終: train {last['train_acc']:.3f} val {last['val_acc']:.3f} test {r['test_acc']:.3f}")


if __name__ == "__main__":
    main()
