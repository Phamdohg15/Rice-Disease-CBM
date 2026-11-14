"""Kiem tra phan xu ly cua server KHONG qua web: du doan 5 anh test -> tao JSON y het server tra ve.
In loi chi tiet neu co. Chay: 4_kiem_tra_server.bat
"""
import faulthandler
import json
import sys
import traceback
from pathlib import Path

faulthandler.enable()
sys.stdout.reconfigure(encoding="utf-8")
sys.argv = [sys.argv[0]]
APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
print("Nap server...", flush=True)
import server  # noqa: E402  (nap mo hinh nhu khi chay that)

I = Path(r"D:\Thesis\Dataset\Data\images\test")
files = [next((I / c).glob("*.jpg")) for c in ["01_dao_on", "02_dom_nau", "03_bac_la", "04_kho_van", "05_la_khoe"]]
for f in files:
    print("=" * 60, "\n", f.parent.name, f.name[:24], flush=True)
    try:
        from PIL import Image
        r = server.PRED.predict(Image.open(f))
        print("  predict OK:", r["ket_luan"], "| ban_do:", None if r["ban_do_chu_y"] is None else r["ban_do_chu_y"].shape, flush=True)
        resp = server._response(r)
        print("  _response OK", flush=True)
        s = json.dumps(resp, ensure_ascii=False, allow_nan=False)
        print(f"  JSON OK, {len(s) / 1024:.0f} KB", flush=True)
    except Exception:
        traceback.print_exc()
print("XONG")
