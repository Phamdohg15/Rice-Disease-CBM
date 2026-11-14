"""Buoc 1: du doan 1 hay nhieu anh tu dong lenh, in ket qua va luu heatmap.

  python du_doan_1_anh.py duong_dan_anh.jpg [anh2.jpg ...]   (hoac 1 thu muc)
  python du_doan_1_anh.py ... --cpu        (ep chay CPU de do toc do nhu tren may khong co GPU)
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
from core.predictor import Predictor  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anh", nargs="+")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=str(APP / "ket_qua_thu"))
    a = ap.parse_args()
    files = []
    for p in map(Path, a.anh):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXT) if p.is_dir() else [p]
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    pr = Predictor(device="cpu" if a.cpu else "auto")
    print(f"Mo hinh: {pr.cfg_app['model_name']} | thiet bi: {pr.device.type} | nap mo hinh {pr.load_seconds:.1f} s")
    print(f"Nguong 'Khong chac chan': {pr.uncertain_threshold}\n")
    times = []
    for f in files:
        r = pr.predict(f)
        times.append(r["thoi_gian_tong_ms"])
        print("=" * 70)
        print(f"Anh: {f.name}")
        print(f"KET LUAN: {r['ket_luan']}")
        print("  (xac suat, chi de kiem tra: " + ", ".join(f"{k} {v:.2f}" for k, v in r["xac_suat_benh"].items()) + ")")
        print("Dau hieu dac trung tren anh:")
        for s in r["dau_hieu"]:
            print(f"  - {s['ten']} ({s['ma']}) - do tin cay {s['do_tin_cay'] * 100:.0f}%")
        if not r["dau_hieu"]:
            print("  (khong co)")
        print(f"Ontology [{r['ontology']['trang_thai']}]: {r['ontology']['thong_diep']}")
        hp = out / f"{f.stem}_heatmap.png"
        r["heatmap"].save(hp)
        print(f"Heatmap: {hp}  ({r['ghi_chu_heatmap']})")
        print(f"Thoi gian: mo hinh {r['thoi_gian_mo_hinh_ms']:.0f} ms, tong {r['thoi_gian_tong_ms']:.0f} ms")
    if times:
        print("=" * 70)
        print(f"{len(times)} anh, thoi gian tong trung binh {sum(times) / len(times):.0f} ms/anh, lon nhat {max(times):.0f} ms ({pr.device.type})")


if __name__ == "__main__":
    main()
