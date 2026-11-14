"""Chan doan heatmap: so sanh nhieu cach ve 'mo hinh nhin vao dau' tren cung 1 anh.

Moi anh -> 1 hinh luoi:
  hang 1: anh goc | attention tong (cach app dang dung) | attention chi tang thap (stride 16) | chi tang cao (stride 32)
          | Grad-CAM cua benh du doan (tren dac trung backbone, TRUOC Transformer)
  hang 2: attention rieng cua tung concept dang bat
Khong sua mo hinh, chi doc ra.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
from core.predictor import Predictor  # noqa: E402
from models import _resize_pos  # noqa: E402


def forward_full(m, x):
    """Giong ConceptAttentionCBM.forward nhung tra them dac trung backbone + attention 2 tang rieng."""
    f = m.body(x)
    lo, hi = m.proj_low(f["low"]), m.proj_high(f["high"])
    b, _, h_lo, w_lo = lo.shape
    h_hi, w_hi = hi.shape[2:]
    t_lo = lo.flatten(2).transpose(1, 2) + _resize_pos(m.pos_low, m.grid_low, (h_lo, w_lo)) + m.scale_emb[0]
    t_hi = hi.flatten(2).transpose(1, 2) + _resize_pos(m.pos_high, m.grid_high, (h_hi, w_hi)) + m.scale_emb[1]
    tokens = m.norm_tokens(m.encoder(torch.cat([t_lo, t_hi], 1)))
    q = m.queries.unsqueeze(0).expand(b, -1, -1)
    z, attn = m.cross(q, tokens, tokens, need_weights=True, average_attn_weights=True)
    z = m.norm_q(z + q)
    cl = (z * m.concept_w).sum(-1) + m.concept_b
    logits = m.task_head(torch.sigmoid(cl))
    n = h_lo * w_lo
    a_lo = attn[:, :, :n].reshape(b, -1, h_lo, w_lo)
    a_hi = attn[:, :, n:].reshape(b, -1, h_hi, w_hi)
    return f, logits, cl, a_lo, a_hi


def gradcam(feat, score):
    g = torch.autograd.grad(score, feat, retain_graph=True)[0]
    w = g.mean((2, 3), keepdim=True)
    return F.relu((w * feat).sum(1))[0]


def to_img(m, w, h):
    """heatmap tren anh da dem vuong -> cat ve kich thuoc anh hien thi, chuan hoa 0..1"""
    s = max(w, h)
    t = F.interpolate(m[None, None].float(), size=(s, s), mode="bilinear", align_corners=False)[0, 0]
    t = t[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w].detach().cpu().numpy()
    return (t - t.min()) / (t.max() - t.min() + 1e-8)


def main():
    pr = Predictor()
    m = pr.model
    root = Path(pr.cfg["data_root"])
    man = pd.read_csv(root / "manifest.csv")
    t = man[man.split == "test"]
    picks = [g.output_relative_path.iloc[0] for _, g in t.groupby("class_name")]
    picks += t[t.class_name == "02_dom_nau"].output_relative_path.iloc[1:7].tolist()
    out = APP / "ket_qua_thu" / "chan_doan_heatmap"
    out.mkdir(parents=True, exist_ok=True)
    for rel in picks:
        p = root / "images" / rel
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        disp = pr._display(im)
        w, h = disp.size
        x = pr.tf(im).unsqueeze(0).to(pr.device)
        with torch.enable_grad():
            f, logits, cl, a_lo, a_hi = forward_full(m, x)
            k = int(logits.argmax())
            cams = {nm: gradcam(f[nm], logits[0, k]) for nm in ("low", "high")}
        cp = torch.sigmoid(cl)[0].detach().cpu().numpy()
        active = [c for i, c in enumerate(pr.concepts) if cp[i] >= pr.thresholds[c] and c != pr.absence]
        idx = [pr.concepts.index(c) for c in active]
        a_hi_up = F.interpolate(a_hi.float(), size=a_lo.shape[2:], mode="bilinear", align_corners=False)
        both = (a_lo.float() + a_hi_up)[0]
        if idx:
            wts = torch.tensor(cp[idx], device=both.device)[:, None, None]
            tot = (both[idx] * wts).sum(0)
            lo_only = (a_lo[0, idx].float() * wts).sum(0)
            hi_only = (a_hi[0, idx].float() * wts).sum(0)
        else:
            tot = lo_only = hi_only = torch.zeros(a_lo.shape[2:])
        top = [("Ảnh gốc", None), ("Attention tổng (app)", tot), ("Attention tầng thấp", lo_only),
               ("Attention tầng cao", hi_only), (f"Grad-CAM {pr.classes[k]} (tầng cao)", cams["high"]),
               ("Grad-CAM (tầng thấp)", cams["low"])]
        ncol = max(len(top), len(idx))
        fig, ax = plt.subplots(2, ncol, figsize=(3 * ncol, 6.6))
        for a in ax.ravel():
            a.axis("off")
        for j, (ttl, mp) in enumerate(top):
            ax[0, j].imshow(disp)
            if mp is not None:
                ax[0, j].imshow(to_img(mp, w, h), cmap="jet", alpha=0.45)
            ax[0, j].set_title(ttl, fontsize=8)
        for j, i in enumerate(idx):
            ax[1, j].imshow(disp)
            ax[1, j].imshow(to_img(both[i], w, h), cmap="jet", alpha=0.45)
            ax[1, j].set_title(f"{pr.concepts[i]} p={cp[i]:.2f}", fontsize=8)
        true = rel.split("/")[1]
        fig.suptitle(f"{Path(rel).name[:20]}  thật: {true}  đoán: {pr.classes[k]}", fontsize=9)
        fig.tight_layout()
        fp = out / f"{true}_{Path(rel).stem[:16]}.png"
        fig.savefig(fp, dpi=90)
        plt.close(fig)
        print("Đã lưu", fp)


if __name__ == "__main__":
    main()
