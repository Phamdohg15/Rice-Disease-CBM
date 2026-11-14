"""Doc anh + nhan concept V1.8. Tien xu ly giong het baseline cu (Project_Rice_Leaf_CNN/train.py)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent.parent
SPLITS = ["train", "val", "test"]


def load_config(backbone):
    common = json.loads((HERE / "configs" / "common.json").read_text(encoding="utf-8"))
    bb = json.loads((HERE / "configs" / f"{backbone}.json").read_text(encoding="utf-8"))
    cfg = dict(common)
    cfg["backbone"] = bb
    return cfg


class PadSquare:
    def __call__(self, im):
        w, h = im.size
        s = max(w, h)
        return ImageOps.expand(im, ((s - w) // 2, (s - h) // 2, s - w - (s - w) // 2, s - h - (s - h) // 2), fill=(0, 0, 0))


def build_transform(size, train):
    ops = [PadSquare(), T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR, antialias=True)]
    if train:
        ops += [T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_split(cfg, split):
    """Ghep manifest (duong dan anh) voi file nhan FINAL theo image_id."""
    root = Path(cfg["data_root"])
    man = pd.read_csv(root / "manifest.csv")
    man = man[man["split"] == split][["image_id", "class_name", "output_relative_path"]]
    lab = pd.read_csv(Path(cfg["label_dir"]) / cfg["label_files"][split])
    lab = lab[["image_id", "class_name"] + cfg["concepts"]].rename(columns={"class_name": "class_label_file"})
    df = man.merge(lab, on="image_id", how="outer", validate="one_to_one", indicator=True)
    bad = df[df["_merge"] != "both"]
    if len(bad):
        raise ValueError(f"[{split}] {len(bad)} anh khong khop giua manifest va file nhan: {bad['image_id'].head().tolist()}")
    wrong = df[df["class_name"] != df["class_label_file"]]
    if len(wrong):
        raise ValueError(f"[{split}] {len(wrong)} anh lech lop giua manifest va file nhan")
    df = df.drop(columns=["_merge", "class_label_file"]).sort_values("image_id").reset_index(drop=True)
    df["path"] = [str(root / "images" / p) for p in df["output_relative_path"]]
    df["y"] = [cfg["classes"].index(c) for c in df["class_name"]]
    for c in cfg["concepts"]:
        df[c] = df[c].astype(int)
    return df


def concept_matrix(df, cfg):
    return df[cfg["concepts"]].to_numpy(dtype=np.float32)  # gia tri 0/1/9


def pos_weight_from_train(train_df, cfg):
    c = concept_matrix(train_df, cfg)
    n1 = (c == 1).sum(0)
    n0 = (c == 0).sum(0)
    pw = np.clip(n0 / np.maximum(n1, 1), None, cfg["cbm"]["pos_weight_max"])
    return torch.tensor(pw, dtype=torch.float32)


class RiceConceptDataset(Dataset):
    def __init__(self, df, cfg, train):
        self.paths = df["path"].tolist()
        self.y = df["y"].tolist()
        self.c = concept_matrix(df, cfg)
        self.tf = build_transform(cfg["image_size"], train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(ImageOps.exif_transpose(im).convert("RGB"))
        return x, self.y[i], torch.from_numpy(self.c[i]), i


def make_loader(df, cfg, train):
    return DataLoader(RiceConceptDataset(df, cfg, train), batch_size=cfg["backbone"]["batch_size"], shuffle=train,
                      num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False)
