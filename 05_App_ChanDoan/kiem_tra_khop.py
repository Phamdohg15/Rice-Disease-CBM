"""Kiem tra app cho ra DUNG ket qua nhu luc danh gia mo hinh (CBM_V18/runs/.../test_predictions.csv).

Chay toan bo 126 anh test qua Predictor cua app roi so sanh:
  - lop du doan phai trung khop
  - xac suat benh / concept chi duoc lech rat nho (luc danh gia dung fp16 tren GPU, app dung fp32)
Ghi them: ty le 'Khong chac chan', thong ke trang thai ontology, thoi gian moi anh.
Chi dung o may ca nhan (can thu muc Dataset va CBM_V18), khong can cho ban dong goi.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
from core.predictor import Predictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--split", default="test")
ap.add_argument("--run_dir", default=r"D:\Thesis\CBM_V18\runs\convnext_tiny_cbm_seed43")
a = ap.parse_args()

pr = Predictor(device="cpu" if a.cpu else "auto")
ref = pd.read_csv(Path(a.run_dir) / f"{a.split}_predictions.csv")
root = Path(pr.cfg["data_root"])
man = pd.read_csv(root / "manifest.csv").set_index("image_id")
rows = []
for i, r in ref.iterrows():
    out = pr.predict(root / "images" / man.loc[r.image_id, "output_relative_path"])
    pb = np.array(list(out["xac_suat_benh"].values()))
    pref = r[[f"p_{c}" for c in pr.classes]].to_numpy(float)
    cref = r[[f"cp_{c}" for c in pr.concepts]].to_numpy(float)
    cp = np.array([out["xac_suat_concept"][c] for c in pr.concepts])
    rows.append({"image_id": r.image_id, "true": r.true_class, "pred_ref": r.pred_class, "pred_app": out["lop_du_doan"],
                 "lech_p_max": float(np.abs(pb - pref).max()), "lech_cp_max": float(np.abs(cp - cref).max()),
                 "khong_chac_chan": out["khong_chac_chan"], "ontology": out["ontology"]["trang_thai"],
                 "ms": out["thoi_gian_tong_ms"]})
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(ref)}")
d = pd.DataFrame(rows)
(APP / "ket_qua_thu").mkdir(exist_ok=True)
d.to_csv(APP / "ket_qua_thu" / f"kiem_tra_khop_{a.split}_{pr.device.type}.csv", index=False)
same = (d.pred_ref == d.pred_app).mean()
print(f"\n[{a.split}] {len(d)} anh | thiet bi {pr.device.type}")
print(f"Lop du doan trung voi luc danh gia: {same * 100:.1f}%  ({(d.pred_ref != d.pred_app).sum()} anh khac)")
print(f"Lech xac suat benh lon nhat: {d.lech_p_max.max():.4f} | lech xac suat concept lon nhat: {d.lech_cp_max.max():.4f}")
print(f"Accuracy cua app: {(d.true == d.pred_app).mean():.4f}")
print(f"Khong chac chan (nguong {pr.uncertain_threshold}): {d.khong_chac_chan.sum()} anh")
print("Ontology:", d.ontology.value_counts().to_dict())
print(f"Thoi gian: trung binh {d.ms.mean():.0f} ms/anh, lon nhat {d.ms.max():.0f} ms")
print("KET LUAN:", "DAT" if same == 1.0 and d.lech_p_max.max() < 0.02 else "CAN XEM LAI")
