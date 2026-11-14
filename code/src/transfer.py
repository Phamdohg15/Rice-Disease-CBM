"""Cong cu cho thi nghiem transfer learning (dong bang / mo backbone, dong bang BatchNorm, vong train theo giai doan).
Khong sua train.py / models.py: app va cac luot cu van dung nguyen code cu."""
import time

import pandas as pd
import torch
from torch import nn

from src.train import run_epoch, seed_all, sync

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def split_parts(model, mode, n_classes):
    """Tra ve (tien to backbone, tap ten tham so phan dau moi).
    baseline: backbone = net.*, phan dau = lop Linear cuoi da thay (n_classes dau ra).
    cbm     : backbone = body.*, phan dau = moi tham so con lai."""
    if mode == "cbm":
        head = {n for n, _ in model.named_parameters() if not n.startswith("body.")}
        return "body.", head
    heads = [name for name, m in model.net.named_modules() if isinstance(m, nn.Linear) and m.out_features == n_classes]
    if len(heads) != 1:
        raise RuntimeError(f"Khong xac dinh duoc lop Linear cuoi cua baseline: {heads}")
    pre = f"net.{heads[0]}."
    head = {n for n, _ in model.named_parameters() if n.startswith(pre)}
    return "net.", head


def backbone_bn_modules(model, prefix):
    root = model.body if prefix == "body." else model.net
    return [m for m in root.modules() if isinstance(m, BN_TYPES)]


def bn_param_names(model, prefix):
    root_name = prefix.rstrip(".")
    root = getattr(model, root_name)
    names = set()
    for mname, m in root.named_modules():
        if isinstance(m, BN_TYPES):
            for pname, _ in m.named_parameters(recurse=False):
                names.add(f"{root_name}.{mname}.{pname}")
    return names


def keep_bn_eval(model, bn_modules):
    """Moi lan model.train() duoc goi (trong run_epoch), dua BatchNorm cua backbone ve eval:
    dung thong ke ImageNet, khong cap nhat running mean/var theo batch 8 anh."""
    orig_train = model.train

    def train(mode=True):
        orig_train(mode)
        if mode:
            for m in bn_modules:
                m.eval()
        return model

    model.train = train


def configure_stage(model, mode, n_classes, stage, unfreeze_prefixes, freeze_bn):
    """stage 1: chi phan dau hoc. stage 2: phan dau + cac khoi stride-32 hoc (BatchNorm van dong bang).
    Tra ve dict thong ke de ghi log/metrics."""
    prefix, head = split_parts(model, mode, n_classes)
    bn_names = bn_param_names(model, prefix) if freeze_bn else set()
    full = [prefix + p.rstrip(".") for p in unfreeze_prefixes] if stage == 2 else []
    matched = {p: 0 for p in full}   # so tham so duoc mo (khong tinh BatchNorm)
    found = {p: 0 for p in full}     # so tham so khop ten (de bat loi go sai ten khoi)
    for n, p in model.named_parameters():
        train = n in head
        if not train and stage == 2:
            for f in full:
                if n == f or n.startswith(f + "."):
                    found[f] += p.numel()
                    if n not in bn_names:
                        train = True
                        matched[f] += p.numel()
                    break
        p.requires_grad_(train)
    empty = [f for f, k in found.items() if k == 0]
    if empty:
        raise RuntimeError(f"Cac khoi can mo khong khop tham so nao (sai ten?): {empty}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_head = sum(p.numel() for n, p in model.named_parameters() if n in head)
    return {"trainable_params": n_train, "head_params": n_head, "unfrozen_backbone_params": n_train - n_head,
            "total_params": sum(p.numel() for p in model.parameters()), "unfrozen_blocks": matched}


def fit_stage(model, train_dl, val_dl, crit, cfg, seed, device, out_dir, log, lr, epochs, patience, tag, epoch_offset=0):
    """Giong fit() trong train.py (AdamW, cosine, AMP, tich luy gradient, clip, chon best theo (val F1, -val CE),
    early stopping) nhung chi toi uu cac tham so requires_grad, mot learning rate."""
    bb = cfg["backbone"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=bb["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    hist, best, stale, best_epoch = [], (-1.0, -float("inf")), 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        seed_all(seed + epoch_offset + epoch)
        t = time.perf_counter()
        tr, _ = run_epoch(model, train_dl, device, crit, opt, scaler, bb["accumulation_steps"], bb["grad_clip"])
        va, _ = run_epoch(model, val_dl, device, crit)
        sync(device)
        h = {"stage": tag, "epoch": epoch, "lr": opt.param_groups[0]["lr"], "seconds": time.perf_counter() - t}
        h.update({"train_" + k: v for k, v in tr.items()})
        h.update({"val_" + k: v for k, v in va.items()})
        hist.append(h)
        pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
        score = (va["f1"], -va["ce"])
        if score > best:
            best, stale, best_epoch = score, 0, epoch
            tmp = out_dir / "best.pt.tmp"
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val": va, "stage": tag}, tmp)
            tmp.replace(out_dir / "best.pt")
            mark = " *"
        else:
            stale += 1
            mark = ""
        extra = f" val_cAUC={va['concept_auc']:.3f}" if "concept_auc" in va else ""
        log(f"[{tag}] epoch {epoch:02d} | train loss={tr['loss']:.4f} F1={tr['f1']:.4f} | val CE={va['ce']:.4f} "
            f"acc={va['acc']:.4f} F1={va['f1']:.4f}{extra} | {h['seconds']:.0f}s{mark}")
        sched.step()
        if stale >= patience:
            log(f"[{tag}] Early stopping sau {stale} epoch khong cai thien")
            break
    sync(device)
    return {"epochs_run": len(hist), "best_epoch": best_epoch, "best_val_f1": best[0],
            "training_seconds": time.perf_counter() - start,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else None}, hist
