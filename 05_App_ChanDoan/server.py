"""Server web cua app chan doan (FastAPI). Chay: python server.py  -> tu mo trinh duyet.

Chi chay tren may nay (127.0.0.1), khong can internet.
Anh gui len duoi dang byte tho (khong can python-multipart).
"""
import base64
import faulthandler
import traceback
import io
import socket
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
(APP / "ket_qua_thu").mkdir(exist_ok=True)
_CRASH_LOG = open(APP / "ket_qua_thu" / "server_crash_log.txt", "w", encoding="utf-8")
faulthandler.enable(file=_CRASH_LOG)  # neu Python bi sap (loi C/C++), ghi vet loi vao file nay
from core.predictor import Predictor  # noqa: E402

MAX_BYTES = 30 * 1024 * 1024
print("Dang nap mo hinh...")
PRED = Predictor()
print(f"Da nap mo hinh ({PRED.device.type}).")
LOCK = threading.Lock()  # Predictor giu trong so attention qua hook -> moi lan chi du doan 1 anh

app = FastAPI(title="Chan doan benh la lua")


def _b64(im, quality=90):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _b64_map(m):
    """Ban do chu y 0..1 -> PNG xam (khong nen mat du lieu) de trang web tu pha mau theo thanh truot."""
    buf = io.BytesIO()
    img = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))  # mang 2 chieu uint8 -> anh xam
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return FileResponse(APP / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.post("/api/du_doan")
async def du_doan(request: Request):
    data = await request.body()
    if not data:
        return JSONResponse({"loi": "Chưa có ảnh."}, status_code=400)
    if len(data) > MAX_BYTES:
        return JSONResponse({"loi": "Ảnh quá lớn (tối đa 30 MB)."}, status_code=400)
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except (UnidentifiedImageError, OSError):
        return JSONResponse({"loi": "Không đọc được tệp này. Hãy chọn ảnh JPG hoặc PNG."}, status_code=400)
    try:
        with LOCK:
            r = PRED.predict(im)
        return _response(r)
    except Exception as e:  # loi bat ngo: in chi tiet ra cua so den + bao len trang, khong lam sap server
        traceback.print_exc()
        return JSONResponse({"loi": f"Lỗi khi dự đoán: {type(e).__name__}: {e}"}, status_code=500)


def _response(r):
    has_heat = bool(r["dau_hieu"]) and any(s["ma"] != PRED.absence for s in r["dau_hieu"])
    return {
        "ket_luan": r["ket_luan"],
        "khong_chac_chan": r["khong_chac_chan"],
        "dau_hieu": [{"ma": s["ma"], "ten": s["ten"], "phan_tram": round(100 * s["do_tin_cay"])} for s in r["dau_hieu"]],
        "dau_hieu_co_the": [{"ma": s["ma"], "ten": s["ten"], "phan_tram": round(100 * s["do_tin_cay"])}
                            for s in r["dau_hieu_co_the"]],
        "ontology": {"trang_thai": r["ontology"]["trang_thai"], "thong_diep": r["ontology"]["thong_diep"]},
        "anh_goc": _b64(r["anh_hien_thi"]),
        "heatmap": _b64(r["heatmap"]) if has_heat else None,
        "ban_do": _b64_map(r["ban_do_chu_y"]) if has_heat and r["ban_do_chu_y"] is not None else None,
        "do_dam_mac_dinh": float(PRED.cfg_app.get("heatmap_alpha", 0.6)),
        "ghi_chu_heatmap": r["ghi_chu_heatmap"],
    }


def _free_port(start=8000, end=8020):
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("Khong tim duoc cong trong (8000-8019)")


if __name__ == "__main__":
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"Mo trinh duyet tai {url}  (dong cua so nay de tat app)")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
