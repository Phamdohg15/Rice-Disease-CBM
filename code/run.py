"""Chay DUNG 1 luot: 1 backbone, 1 loai (baseline/cbm), 1 seed.
Vi du: python run.py --backbone resnet18 --mode cbm --seed 42
Ket qua: runs/<backbone>_<mode>_seed<seed>/ ; metrics.json ghi CUOI CUNG = dau hieu luot da xong.
Neu da co metrics.json thi bo qua (tru khi --overwrite).
Ablation 16 concept (them C08, C13 vao bottleneck): them --concept-set 16 -> ket qua o runs_16c/
(mac dinh --concept-set 14 = dung y nhu cac luot da chay trong runs/)."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import gc
import json
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

from src.data import load_config, load_split, make_loader, pos_weight_from_train
from src.evaluate import choose_thresholds, concept_table, predictions_table, rule_violation_rate, task_report
from src.losses import Criterion
from src.models import build_model
from src.train import fit, run_epoch, seed_all, sync


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    tmp.replace(path)


def plot_history(hist, out):
    h = pd.DataFrame(hist)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(h["epoch"], h["train_loss"], label="train loss")
    axs[0].plot(h["epoch"], h["val_loss"], label="val loss")
    axs[1].plot(h["epoch"], h["train_f1"], label="train F1")
    axs[1].plot(h["epoch"], h["val_f1"], label="val F1")
    if "val_concept_auc" in h:
        axs[1].plot(h["epoch"], h["val_concept_auc"], label="val concept AUC")
    for ax in axs:
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["resnet18", "densenet121", "efficientnet_b0", "convnext_tiny", "mobilenet_v2"])
    ap.add_argument("--mode", required=True, choices=["baseline", "cbm"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, help="Chi dung de chay thu nhanh")
    ap.add_argument("--runs-root", help="Mac dinh lay tu configs/common.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--concept-set", choices=["14", "16"], default="14",
                    help="14 = bottleneck da chot (mac dinh); 16 = ablation them C08, C13")
    args = ap.parse_args()

    cfg = load_config(args.backbone)
    if args.epochs:
        cfg["backbone"]["epochs"] = args.epochs
    if args.concept_set == "16":
        if args.mode != "cbm":
            print("--concept-set 16 chi dung cho --mode cbm (baseline khong dung concept)")
            return 1
        cfg["concepts"] = [f"C{i:02d}" for i in range(1, 17)]  # tham so khac giu nguyen
        default_root = HERE / "runs_16c"
    else:
        default_root = Path(cfg["runs_root"])
    root = Path(args.runs_root) if args.runs_root else default_root
    out = root / f"{args.backbone}_{args.mode}_seed{args.seed}"
    if (out / "metrics.json").exists() and not args.overwrite:
        old_cbm = json.loads((out / "config.json").read_text(encoding="utf-8")).get("cbm")
        if args.mode == "cbm" and old_cbm != cfg["cbm"]:
            # Luot CBM cu chay voi tham so CBM khac -> giu lai trong runs_cu/, khong xoa
            dst = root.parent / "runs_cu" / f"{out.name}_{datetime.now():%Y%m%d_%H%M%S}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out), str(dst))
            print(f"Tham so CBM da doi -> chuyen luot cu sang {dst} va chay lai")
        else:
            print(f"DA CHAY ROI, bo qua: {out.name}")
            return 0
    flag = root / "TAM_HOAN_CBM.txt"
    if args.mode == "cbm" and flag.exists() and "TAM_HOAN=1" in flag.read_text(encoding="utf-8", errors="ignore"):
        print(f"TAM HOAN (TAM_HOAN_CBM.txt co TAM_HOAN=1), bo qua: {out.name}")
        return 0
    if out.exists():
        shutil.rmtree(out)  # luot dang do hoac --overwrite
    out.mkdir(parents=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Khong thay GPU CUDA")
        device = torch.device("cuda")
        cfg["run"] = {"backbone": args.backbone, "mode": args.mode, "seed": args.seed, "concept_set": args.concept_set, "started": datetime.now().isoformat(timespec="seconds"),
                      "env": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                              "gpu": torch.cuda.get_device_name(0)}}
        write_json(out / "config.json", cfg)
        log(f"=== {out.name} | GPU {torch.cuda.get_device_name(0)} | {len(cfg['concepts'])} concept ===")

        dfs = {s: load_split(cfg, s) for s in ["train", "val", "test"]}
        log("So anh: " + ", ".join(f"{s}={len(d)}" for s, d in dfs.items()))
        pos_weight = pos_weight_from_train(dfs["train"], cfg)
        seed_all(args.seed)
        model = build_model(cfg, args.mode).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"Tham so: {n_params:,}")
        crit = Criterion(cfg, args.mode, pos_weight, device)
        train_dl = make_loader(dfs["train"], cfg, True)
        val_dl = make_loader(dfs["val"], cfg, False)
        test_dl = make_loader(dfs["test"], cfg, False)

        info, hist = fit(model, train_dl, val_dl, crit, cfg, args.seed, device, out, log)
        plot_history(hist, out)
        log(f"Train xong: {info['epochs_run']} epoch, best epoch {info['best_epoch']}, {info['training_seconds'] / 60:.1f} phut")

        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        res = {"backbone": args.backbone, "mode": args.mode, "seed": args.seed, "concept_set": args.concept_set,
               "n_concepts": len(cfg["concepts"]), "parameters": n_params, **info}
        for split, dl in [("val", val_dl), ("test", test_dl)]:
            sync(device)
            t = time.perf_counter()
            m, data = run_epoch(model, dl, device, crit)
            sync(device)
            res[f"{split}_acc"], res[f"{split}_f1"], res[f"{split}_ce"] = m["acc"], m["f1"], m["ce"]
            if split == "test":
                res["test_ms_per_image"] = 1000 * (time.perf_counter() - t) / len(dl.dataset)
            rep, cm = task_report(data["y"], data["prob"], cfg["classes"])
            rep.to_csv(out / f"{split}_classification_report.csv")
            cm.to_csv(out / f"{split}_confusion_matrix.csv")
            predictions_table(dfs[split], data, cfg).to_csv(out / f"{split}_predictions.csv", index=False)
            if args.mode == "cbm":
                if split == "val":
                    th = choose_thresholds(data["cprob"], data["clab"], cfg["concepts"])
                    write_json(out / "concept_thresholds.json", th)
                ct = concept_table(data["cprob"], data["clab"], cfg["concepts"], th, cfg["cbm"]["auc_ontology_min"])
                ct.to_csv(out / f"{split}_concept_metrics.csv", index=False)
                res[f"{split}_concept_auc"] = float(ct["auc"].mean())
                res[f"{split}_concept_f1"] = float(ct["f1"].mean())
                res[f"{split}_concept_acc"] = float(ct["acc"].mean())
                res[f"{split}_n_concept_auc_ge_min"] = int(ct["dat_nguong_ontology"].sum())
                res[f"{split}_rule_violation"], res[f"{split}_n_pred_C16"] = rule_violation_rate(
                    data["cprob"], cfg["concepts"], th, cfg["absence_concept"])
            log(f"{split.upper()}: acc={m['acc']:.4f} F1={m['f1']:.4f}" +
                (f" concept AUC={res[f'{split}_concept_auc']:.3f} F1={res[f'{split}_concept_f1']:.3f}" if args.mode == "cbm" else ""))
        res["finished"] = datetime.now().isoformat(timespec="seconds")
        write_json(out / "metrics.json", res)
        log("XONG")
        return 0
    except Exception:
        tb = traceback.format_exc()
        log("LOI:\n" + tb)
        (out / "error.txt").write_text(tb, encoding="utf-8")
        return 1
    finally:
        logf.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
