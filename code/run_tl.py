"""Thi nghiem transfer learning - chay DUNG 1 luot: 1 backbone, 1 loai (baseline/cbm), 1 seed, 1 chien luoc.

Chien luoc (A = fine-tune toan bo tu ImageNet: da co trong runs/, khong chay lai o day):
  --strategy two_stage : B. Hai giai doan theo notebook cua thay.
                         GD1: dong bang backbone, hoc phan dau (lr 1e-3) -> luu thanh luot "frozen" (= chien luoc D)
                         GD2: tu best GD1, mo khoi stride-32, BatchNorm dong bang, lr 1e-5 -> luot "two_stage"
  --strategy scratch   : C. Khong dung trong so ImageNet, moi thu khac giong het A.

Vi du: python run_tl.py --backbone resnet18 --mode cbm --seed 42 --strategy two_stage
Ket qua: runs_tl/<backbone>_<mode>_<frozen|two_stage|scratch>_seed<seed>/ ; metrics.json ghi cuoi = luot da xong.
Luot da xong thi bo qua (tru khi --overwrite). Neu GD1 da xong ma GD2 chua, chay lai chi lam GD2."""
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
import pandas as pd
import torch
import torchvision

from src.data import load_config, load_split, make_loader, pos_weight_from_train
from src.evaluate import choose_thresholds, concept_table, predictions_table, rule_violation_rate, task_report
from src.losses import Criterion
from src.models import build_model
from src.train import fit, run_epoch, seed_all, sync
from src.transfer import configure_stage, backbone_bn_modules, fit_stage, keep_bn_eval, split_parts

BACKBONES = ["resnet18", "densenet121", "efficientnet_b0", "convnext_tiny", "mobilenet_v2"]


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    tmp.replace(path)


def plot_history(hist, out, title):
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
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)


class Run:
    """Mot thu muc ket qua: log rieng, config rieng, metrics.json ghi cuoi."""

    def __init__(self, out):
        self.out = out
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        self.f = open(out / "log.txt", "a", encoding="utf-8")

    def log(self, msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def evaluate(model, run, ctx, res):
    """Danh gia best.pt tren val va test, ghi cac file giong het run.py."""
    cfg, mode, device, crit = ctx["cfg"], ctx["mode"], ctx["device"], ctx["crit"]
    out = run.out
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    th = None
    for split in ("val", "test"):
        dl = ctx["dls"][split]
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
        predictions_table(ctx["dfs"][split], data, cfg).to_csv(out / f"{split}_predictions.csv", index=False)
        if mode == "cbm":
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
        run.log(f"{split.upper()}: acc={m['acc']:.4f} F1={m['f1']:.4f}" +
                (f" concept AUC={res[f'{split}_concept_auc']:.3f} F1={res[f'{split}_concept_f1']:.3f}" if mode == "cbm" else ""))
    res["finished"] = datetime.now().isoformat(timespec="seconds")
    write_json(out / "metrics.json", res)
    run.log("XONG")


def base_res(ctx, strategy, model):
    return {"backbone": ctx["backbone"], "mode": ctx["mode"], "seed": ctx["seed"], "strategy": strategy,
            "parameters": sum(p.numel() for p in model.parameters())}


def save_config(run, ctx, strategy, extra):
    cfg = dict(ctx["cfg"])
    cfg["transfer"] = ctx["tl"]
    cfg["run"] = {"backbone": ctx["backbone"], "mode": ctx["mode"], "seed": ctx["seed"], "strategy": strategy,
                  "started": datetime.now().isoformat(timespec="seconds"), **extra,
                  "env": {"python": platform.python_version(), "torch": torch.__version__,
                          "torchvision": torchvision.__version__, "gpu": torch.cuda.get_device_name(0)}}
    write_json(run.out / "config.json", cfg)


def stage_model(ctx, stage, init_state=None):
    seed_all(ctx["seed"])
    model = build_model(ctx["cfg"], ctx["mode"], pretrained=True)
    if init_state is not None:
        model.load_state_dict(init_state)
    model = model.to(ctx["device"])
    tl = ctx["tl"]
    info = configure_stage(model, ctx["mode"], len(ctx["cfg"]["classes"]), stage,
                           tl["unfreeze_stride32"][ctx["backbone"]], tl["freeze_batchnorm"])
    if tl["freeze_batchnorm"] or stage == 1:
        prefix, _ = split_parts(model, ctx["mode"], len(ctx["cfg"]["classes"]))
        keep_bn_eval(model, backbone_bn_modules(model, prefix))
    return model, info


def run_two_stage(ctx, root, overwrite):
    bb, mode, seed, tl = ctx["backbone"], ctx["mode"], ctx["seed"], ctx["tl"]
    d1 = root / f"{bb}_{mode}_frozen_seed{seed}"
    d2 = root / f"{bb}_{mode}_two_stage_seed{seed}"
    if (d2 / "metrics.json").exists() and not overwrite:
        print(f"DA CHAY ROI, bo qua: {d2.name}")
        return
    # ---------------- Giai doan 1 (= chien luoc D "frozen")
    if not (d1 / "metrics.json").exists() or overwrite:
        run = Run(d1)
        try:
            model, info = stage_model(ctx, 1)
            save_config(run, ctx, "frozen", {"stage": 1, "stage_info": info, "stage_settings": tl["stage1"]})
            run.log(f"=== {d1.name} | GD1: dong bang backbone, hoc phan dau | lr={tl['stage1']['learning_rate']} | "
                    f"tham so hoc {info['trainable_params']:,}/{info['total_params']:,} ===")
            fi, hist = fit_stage(model, ctx["dls"]["train"], ctx["dls"]["val"], ctx["crit"], ctx["cfg"], seed, ctx["device"],
                                 d1, run.log, tl["stage1"]["learning_rate"], ctx["epochs1"], tl["stage1"]["patience"], "GD1")
            plot_history(hist, d1, f"{d1.name} (GD1: dong bang backbone)")
            run.log(f"GD1 xong: {fi['epochs_run']} epoch, best epoch {fi['best_epoch']}, {fi['training_seconds'] / 60:.1f} phut")
            res = base_res(ctx, "frozen", model)
            res.update(fi)
            res.update({"trainable_params": info["trainable_params"], "unfrozen_backbone_params": 0})
            evaluate(model, run, ctx, res)
        except Exception:
            tb = traceback.format_exc()
            run.log("LOI:\n" + tb)
            (d1 / "error.txt").write_text(tb, encoding="utf-8")
            raise
        finally:
            run.close()
    else:
        print(f"GD1 da xong, dung lai: {d1.name}")
    m1 = json.loads((d1 / "metrics.json").read_text(encoding="utf-8"))
    # ---------------- Giai doan 2 (= chien luoc B "two_stage")
    run = Run(d2)
    try:
        state = torch.load(d1 / "best.pt", map_location="cpu", weights_only=False)["state_dict"]
        model, info = stage_model(ctx, 2, state)
        save_config(run, ctx, "two_stage", {"stage": 2, "init_from": str(d1 / "best.pt"), "stage_info": info,
                                            "stage_settings": tl["stage2"]})
        run.log(f"=== {d2.name} | GD2: mo khoi stride-32 {list(info['unfrozen_blocks'])} | BatchNorm dong bang | "
                f"lr={tl['stage2']['learning_rate']} | tham so hoc {info['trainable_params']:,}/{info['total_params']:,} ===")
        fi, hist = fit_stage(model, ctx["dls"]["train"], ctx["dls"]["val"], ctx["crit"], ctx["cfg"], seed, ctx["device"],
                             d2, run.log, tl["stage2"]["learning_rate"], ctx["epochs2"], tl["stage2"]["patience"], "GD2",
                             epoch_offset=1000)
        plot_history(hist, d2, f"{d2.name} (GD2: fine-tune khoi stride-32)")
        run.log(f"GD2 xong: {fi['epochs_run']} epoch, best epoch {fi['best_epoch']}, {fi['training_seconds'] / 60:.1f} phut")
        res = base_res(ctx, "two_stage", model)
        res.update({"stage1_epochs_run": m1["epochs_run"], "stage1_best_epoch": m1["best_epoch"],
                    "stage1_training_seconds": m1["training_seconds"], "stage1_val_f1": m1["val_f1"],
                    "stage2_epochs_run": fi["epochs_run"], "stage2_best_epoch": fi["best_epoch"],
                    "stage2_training_seconds": fi["training_seconds"],
                    "epochs_run": m1["epochs_run"] + fi["epochs_run"], "best_epoch": m1["epochs_run"] + fi["best_epoch"],
                    "best_val_f1": fi["best_val_f1"], "training_seconds": m1["training_seconds"] + fi["training_seconds"],
                    "peak_gpu_memory_mb": fi["peak_gpu_memory_mb"], "trainable_params": info["trainable_params"],
                    "unfrozen_backbone_params": info["unfrozen_backbone_params"]})
        evaluate(model, run, ctx, res)
    except Exception:
        tb = traceback.format_exc()
        run.log("LOI:\n" + tb)
        (d2 / "error.txt").write_text(tb, encoding="utf-8")
        raise
    finally:
        run.close()


def run_scratch(ctx, root, overwrite):
    bb, mode, seed = ctx["backbone"], ctx["mode"], ctx["seed"]
    d = root / f"{bb}_{mode}_scratch_seed{seed}"
    if (d / "metrics.json").exists() and not overwrite:
        print(f"DA CHAY ROI, bo qua: {d.name}")
        return
    run = Run(d)
    try:
        seed_all(seed)
        model = build_model(ctx["cfg"], mode, pretrained=False).to(ctx["device"])
        save_config(run, ctx, "scratch", {"pretrained": False})
        run.log(f"=== {d.name} | khong dung trong so ImageNet, cac thiet lap khac giong luot A | "
                f"tham so {sum(p.numel() for p in model.parameters()):,} ===")
        info, hist = fit(model, ctx["dls"]["train"], ctx["dls"]["val"], ctx["crit"], ctx["cfg"], seed, ctx["device"], d, run.log)
        plot_history(hist, d, f"{d.name} (huan luyen tu dau)")
        run.log(f"Train xong: {info['epochs_run']} epoch, best epoch {info['best_epoch']}, {info['training_seconds'] / 60:.1f} phut")
        res = base_res(ctx, "scratch", model)
        res.update(info)
        res["trainable_params"] = res["parameters"]
        evaluate(model, run, ctx, res)
    except Exception:
        tb = traceback.format_exc()
        run.log("LOI:\n" + tb)
        (d / "error.txt").write_text(tb, encoding="utf-8")
        raise
    finally:
        run.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=BACKBONES)
    ap.add_argument("--mode", required=True, choices=["baseline", "cbm"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--strategy", required=True, choices=["two_stage", "scratch"])
    ap.add_argument("--epochs", type=int, help="Chi dung de chay thu nhanh: gioi han so epoch moi giai doan")
    ap.add_argument("--runs-root", help="Mac dinh lay tu configs/transfer.json (runs_tl)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.backbone)
    tl = json.loads((HERE / "configs" / "transfer.json").read_text(encoding="utf-8"))
    root = Path(args.runs_root or tl["runs_root"])
    if not root.is_absolute():
        root = HERE / root
    root.mkdir(parents=True, exist_ok=True)
    epochs1, epochs2 = tl["stage1"]["epochs"], tl["stage2"]["epochs"]
    if args.epochs:
        cfg["backbone"]["epochs"] = args.epochs
        epochs1 = epochs2 = args.epochs
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Khong thay GPU CUDA")
        device = torch.device("cuda")
        dfs = {s: load_split(cfg, s) for s in ["train", "val", "test"]}
        ctx = {"cfg": cfg, "tl": tl, "backbone": args.backbone, "mode": args.mode, "seed": args.seed, "device": device,
               "dfs": dfs, "epochs1": epochs1, "epochs2": epochs2,
               "crit": Criterion(cfg, args.mode, pos_weight_from_train(dfs["train"], cfg), device),
               "dls": {"train": make_loader(dfs["train"], cfg, True), "val": make_loader(dfs["val"], cfg, False),
                       "test": make_loader(dfs["test"], cfg, False)}}
        if args.strategy == "two_stage":
            run_two_stage(ctx, root, args.overwrite)
        else:
            run_scratch(ctx, root, args.overwrite)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
