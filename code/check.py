"""Giai doan 0: kiem tra du lieu + chay thu mo hinh (chua train).
python check.py               -> kiem tra du lieu + 5 backbone x (baseline, cbm)
python check.py --data-only   -> chi kiem tra du lieu
Ket qua ghi vao check_report.txt"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from PIL import Image

from src.data import load_config, load_split, make_loader, pos_weight_from_train
from src.losses import Criterion
from src.models import build_model

BACKBONES = ["resnet18", "densenet121", "efficientnet_b0", "convnext_tiny", "mobilenet_v2"]
report = open(HERE / "check_report.txt", "w", encoding="utf-8")
n_fail = 0


def say(msg=""):
    print(msg, flush=True)
    report.write(msg + "\n")
    report.flush()


def ok(cond, msg):
    global n_fail
    say(("  [OK]  " if cond else "  [LOI] ") + msg)
    if not cond:
        n_fail += 1


def check_data(cfg):
    say("== 1. DU LIEU ==")
    dfs = {}
    for s in ["train", "val", "test"]:
        df = load_split(cfg, s)
        dfs[s] = df
        ok(len(df) == cfg["expected_counts"][s], f"{s}: {len(df)} anh (mong doi {cfg['expected_counts'][s]})")
        vals = set(np.unique(df[cfg["concepts"]].to_numpy()).tolist())
        ok(vals <= {0, 1, 9}, f"{s}: gia tri nhan concept = {sorted(vals)}")
        n1 = {c: int((df[c] == 1).sum()) for c in cfg["concepts"]}
        diff = {c: (n1[c], cfg["expected_n1"][s][c]) for c in cfg["concepts"] if n1[c] != cfg["expected_n1"][s][c]}
        ok(not diff, f"{s}: so anh nhan 1 tung concept khop bang da chot" + (f" -- LECH: {diff}" if diff else ""))
        c16 = df["C16"] == 1
        lesion = [c for c in cfg["concepts"] if c != "C16"]
        bad = int((df.loc[c16, lesion] == 1).any(axis=1).sum())
        ok(bad == 0, f"{s}: rang buoc C16=1 -> concept ton thuong=0 (vi pham: {bad})")
        missing = [p for p in df["path"] if not Path(p).exists()]
        ok(not missing, f"{s}: file anh ton tai ({len(df) - len(missing)}/{len(df)})" + (f" vd thieu: {missing[:2]}" if missing else ""))
        broken = []
        for p in df["path"]:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception as e:
                broken.append((p, str(e)))
        ok(not broken, f"{s}: mo duoc anh ({len(df) - len(broken)}/{len(df)})" + (f" vd loi: {broken[:2]}" if broken else ""))
        say(f"        so anh moi lop: {df['class_name'].value_counts().sort_index().to_dict()}")
    ids = [set(d["image_id"]) for d in dfs.values()]
    ok(not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "khong co anh trung giua train/val/test")
    pw = pos_weight_from_train(dfs["train"], cfg)
    say("        pos_weight (train): " + ", ".join(f"{c}={w:.2f}" for c, w in zip(cfg["concepts"], pw.tolist())))
    return dfs


def check_models(dfs, backbones):
    say("\n== 2. CHAY THU MO HINH (1 batch anh that, forward + backward, chua train) ==")
    device = torch.device("cuda")
    for name in backbones:
        cfg = load_config(name)
        x, y, c, _ = next(iter(make_loader(dfs["train"], cfg, True)))
        x, y, c = x.to(device), y.to(device), c.to(device)
        pw = pos_weight_from_train(dfs["train"], cfg)
        for mode in ["baseline", "cbm"]:
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.manual_seed(0)
                model = build_model(cfg, mode).to(device).train()
                crit = Criterion(cfg, mode, pw, device)
                t = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x, return_attention=True)
                loss, parts = crit(out, y, c)
                loss.mean().backward()
                torch.cuda.synchronize()
                shapes = {k: tuple(v.shape) for k, v in out.items()}
                grads = sum(p.grad is not None for p in model.parameters())
                n_par = sum(p.numel() for p in model.parameters())
                exp_ok = shapes["class_logits"] == (len(y), 5)
                if mode == "cbm":
                    exp_ok &= shapes["concept_logits"] == (len(y), 14) and shapes["heatmap"][1] == 14
                    extra = f" | kenh {model.channels}, luoi {model.grid_low}+{model.grid_high} | heatmap {shapes['heatmap']}"
                    extra += f" | ce={parts['ce'] / len(y):.3f} bce={parts['bce'] / len(y):.3f} rule={parts['rule'] / len(y):.3f}"
                else:
                    extra = f" | ce={parts['ce'] / len(y):.3f}"
                ok(exp_ok and torch.isfinite(loss).all().item() and grads > 0,
                   f"{name:16s} {mode:8s} | {n_par / 1e6:.2f}M tham so | VRAM {torch.cuda.max_memory_allocated() / 1024 ** 2:.0f}MB"
                   f" | {time.perf_counter() - t:.2f}s{extra}")
            except Exception:
                ok(False, f"{name} {mode}:\n" + traceback.format_exc())
            finally:
                model = None
                gc.collect()
                torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    args = ap.parse_args()
    say(f"Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {torch.cuda.is_available()}"
        + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    cfg = load_config("resnet18")
    dfs = check_data(cfg)
    if not args.data_only:
        if not torch.cuda.is_available():
            ok(False, "Khong thay GPU CUDA")
        else:
            check_models(dfs, args.backbones)
    say(f"\n== KET LUAN: {'TAT CA OK' if n_fail == 0 else f'{n_fail} MUC LOI'} ==")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
