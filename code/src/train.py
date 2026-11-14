"""Vong huan luyen chung cho baseline va CBM. Giu nguyen cach lam cua baseline cu:
AdamW, CosineAnnealing, AMP fp16, tich luy gradient, clip 5, chon best theo (val macro-F1, -val CE), early stopping."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def mean_concept_auc(cprob, clab):
    aucs = []
    for k in range(clab.shape[1]):
        m = clab[:, k] != 9
        yk = clab[m, k]
        if len(np.unique(yk)) == 2:
            aucs.append(roc_auc_score(yk, cprob[m, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


def run_epoch(model, dl, device, crit, optimizer=None, scaler=None, accum=1, grad_clip=5.0, return_attention=False):
    training = optimizer is not None
    model.train(training)
    sums = defaultdict(float)
    ys, probs, cprobs, clabs, ids, heat = [], [], [], [], [], []
    window = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for b, (x, y, c, idx) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x, return_attention=return_attention)
            loss_vec, parts = crit(out, y, c)
            loss = loss_vec.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Loss khong huu han (NaN/Inf)")
            if training:
                scaler.scale(loss).backward()
                window += len(y)
                if (b + 1) % accum == 0 or b + 1 == len(dl):
                    scaler.unscale_(optimizer)
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.div_(window)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    window = 0
            for k, v in parts.items():
                sums[k] += v
            ys.append(y.cpu())
            probs.append(out["class_logits"].detach().float().softmax(1).cpu())
            ids.append(idx)
            clabs.append(c.cpu())
            if "concept_logits" in out:
                cprobs.append(torch.sigmoid(out["concept_logits"].detach().float()).cpu())
            if return_attention and "heatmap" in out:
                heat.append(out["heatmap"].detach().float().cpu())
    n = len(dl.dataset)
    res = {k: v / n for k, v in sums.items()}
    y = torch.cat(ys).numpy()
    prob = torch.cat(probs).numpy()
    pred = prob.argmax(1)
    res["acc"] = float(accuracy_score(y, pred))
    res["f1"] = float(f1_score(y, pred, average="macro", labels=list(range(prob.shape[1])), zero_division=0))
    data = {"y": y, "prob": prob, "idx": torch.cat(ids).numpy(), "clab": torch.cat(clabs).numpy()}
    if cprobs:
        data["cprob"] = torch.cat(cprobs).numpy()
        res["concept_auc"] = mean_concept_auc(data["cprob"], data["clab"])
    if heat:
        data["heatmap"] = torch.cat(heat).numpy()
    return res, data


def fit(model, train_dl, val_dl, crit, cfg, seed, device, out_dir, log):
    bb = cfg["backbone"]
    head_lr = cfg.get("cbm", {}).get("head_learning_rate")
    if head_lr and hasattr(model, "body"):  # CBM: phan moi (chieu, Transformer, truy van, dau concept, MLP) dung lr rieng
        named = list(model.named_parameters())
        params = [{"params": [p for n, p in named if n.startswith("body.")], "lr": bb["learning_rate"]},
                  {"params": [p for n, p in named if not n.startswith("body.")], "lr": head_lr}]
    else:
        params = model.parameters()
    opt = torch.optim.AdamW(params, lr=bb["learning_rate"], weight_decay=bb["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=bb["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    hist, best, stale, best_epoch = [], (-1.0, -float("inf")), 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    start = time.perf_counter()
    for epoch in range(1, bb["epochs"] + 1):
        seed_all(seed + epoch)
        t = time.perf_counter()
        tr, _ = run_epoch(model, train_dl, device, crit, opt, scaler, bb["accumulation_steps"], bb["grad_clip"])
        va, _ = run_epoch(model, val_dl, device, crit)
        sync(device)
        h = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "seconds": time.perf_counter() - t}
        h.update({"train_" + k: v for k, v in tr.items()})
        h.update({"val_" + k: v for k, v in va.items()})
        hist.append(h)
        pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
        score = (va["f1"], -va["ce"])
        if score > best:
            best, stale, best_epoch = score, 0, epoch
            tmp = out_dir / "best.pt.tmp"
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val": va}, tmp)
            tmp.replace(out_dir / "best.pt")
            mark = " *"
        else:
            stale += 1
            mark = ""
        extra = f" val_cAUC={va['concept_auc']:.3f}" if "concept_auc" in va else ""
        log(f"epoch {epoch:02d} | train loss={tr['loss']:.4f} F1={tr['f1']:.4f} | val CE={va['ce']:.4f} "
            f"acc={va['acc']:.4f} F1={va['f1']:.4f}{extra} | {h['seconds']:.0f}s{mark}")
        sched.step()
        if stale >= bb["patience"]:
            log(f"Early stopping sau {stale} epoch khong cai thien")
            break
    sync(device)
    return {"epochs_run": len(hist), "best_epoch": best_epoch, "best_val_f1": best[0],
            "training_seconds": time.perf_counter() - start,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else None}, hist
