"""Tu kiem tra ban dong goi: nap server y nhu khi chay that, du doan 5 anh mau, tao ket qua JSON nhu trang web nhan.
Chay: python\\python.exe kiem_tra_nhanh.py   (dong_goi.ps1 tu chay sau khi dong goi)
"""
import base64
import io
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
sys.argv = [sys.argv[0]]

import torch  # noqa: E402
from PIL import Image  # noqa: E402

print(f"Python {sys.version.split()[0]} tai {sys.executable}")
print(f"torch {torch.__version__} | GPU: {torch.cuda.is_available()}")
t = time.perf_counter()
import server  # noqa: E402  (nap mo hinh nhu khi chay app)
print(f"Nap app + mo hinh: {time.perf_counter() - t:.1f} s, thiet bi {server.PRED.device.type}")

loi = 0
for f in sorted((APP / "anh_mau").glob("*.jpg")):
    t = time.perf_counter()
    r = server.PRED.predict(Image.open(f))
    resp = server._response(r)
    js = json.dumps(resp, ensure_ascii=False, allow_nan=False)
    if resp["ban_do"]:  # anh ban do heatmap phai mo duoc
        Image.open(io.BytesIO(base64.b64decode(resp["ban_do"].split(",", 1)[1]))).load()
    ms = 1000 * (time.perf_counter() - t)
    print(f"  {f.name:16s} -> {resp['ket_luan']:16s} | ontology {resp['ontology']['trang_thai']:13s} | "
          f"heatmap {'co' if resp['ban_do'] else 'khong'} | {ms:.0f} ms")
    if ms > 3000:
        loi += 1
        print("    !! cham hon 3 giay")
print("KET QUA TU KIEM TRA:", "DAT" if loi == 0 else "CAN XEM LAI")
sys.exit(0 if loi == 0 else 1)
