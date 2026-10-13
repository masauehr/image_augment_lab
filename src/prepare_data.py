"""hymenoptera（アリ vs ハチ）データを取得し、train/val/test に分割する。

- 取得元: PyTorch公式チュートリアルのzip（download.pytorch.org）
- 公式の train はそのまま使い、公式の val をクラスごとに val / test へ半分ずつ分ける
- 分割は拡張の「前」に行う（データリーク防止）。シード固定で再現可能
"""
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "hymenoptera"
SEED = 42


def download() -> Path:
    """zipを取得して展開する（取得済みならスキップ）"""
    zip_path = RAW / "hymenoptera_data.zip"
    src_dir = RAW / "hymenoptera_data"
    if src_dir.exists():
        return src_dir
    RAW.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"ダウンロード: {URL}")
        urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        # パストラバーサル対策: 展開先の外を指すエントリは拒否
        for name in zf.namelist():
            if not (RAW / name).resolve().is_relative_to(RAW.resolve()):
                raise ValueError(f"不正なパス: {name}")
        zf.extractall(RAW)
    return src_dir


def split(src_dir: Path) -> None:
    """公式train→train、公式val→val/test（クラスごとに半分ずつ）"""
    if OUT.exists():
        shutil.rmtree(OUT)  # 自分が生成した出力ディレクトリのみ再生成
    rng = random.Random(SEED)
    for cls_dir in sorted((src_dir / "train").iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        # train
        for f in sorted(cls_dir.glob("*.jpg")):
            (OUT / "train" / cls).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, OUT / "train" / cls / f.name)
        # val / test
        files = sorted((src_dir / "val" / cls).glob("*.jpg"))
        rng.shuffle(files)
        half = len(files) // 2
        for part, subset in (("val", files[:half]), ("test", files[half:])):
            (OUT / part / cls).mkdir(parents=True, exist_ok=True)
            for f in subset:
                shutil.copy2(f, OUT / part / cls / f.name)


def summary() -> None:
    for part in ("train", "val", "test"):
        for cls_dir in sorted((OUT / part).iterdir()):
            print(f"{part:5s} {cls_dir.name:6s} {len(list(cls_dir.glob('*.jpg')))}枚")


if __name__ == "__main__":
    split(download())
    summary()
