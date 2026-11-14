"""Loi du doan cua app: 1 anh -> benh, dau hieu (concept), kiem tra ontology, heatmap.

Mo hinh dung lai nguyen ban code CBM_V18/src/models.py (chep sang App/model_src/models.py, khong sua).
Tien xu ly giong het luc train (CBM_V18/src/data.py): xoay theo EXIF -> RGB -> dem vuong (vien den) -> 448x448 -> chuan hoa ImageNet.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms as T

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR / "model_src"))
from models import build_model  # noqa: E402

DISPLAY_MAX_SIDE = 768  # anh hien thi tren web / heatmap
MAYBE_MIN = 0.5  # nguong duoi cua muc "Dau hieu co the co"


def _read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


class PadSquare:
    """Giong CBM_V18/src/data.py: dem anh thanh hinh vuong, anh nam giua, vien den."""

    def __call__(self, im):
        w, h = im.size
        s = max(w, h)
        return ImageOps.expand(im, ((s - w) // 2, (s - h) // 2, s - w - (s - w) // 2, s - h - (s - h) // 2), fill=(0, 0, 0))


def _jet(x):
    """Bang mau jet (xanh -> vang -> do) cho mang 0..1, khong can matplotlib."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], -1)


def blend(base, m, a):
    """Pha heatmap len anh. a = do dam 0..1 (thanh truot tren web dung DUNG cong thuc nay, viet lai bang JS).
    Trong so mau = a * (m + 0.5a): a nho -> chi vung chu y manh co mau; a = 1 -> ca anh phu mau nhu heatmap kinh dien."""
    w = np.clip(a * (m + 0.5 * a), 0, 1)[..., None]
    return (np.clip((1 - w) * base + w * _jet(m), 0, 1) * 255).astype(np.uint8)


class Predictor:
    def __init__(self, app_config="app_config.json", device="auto"):
        cfg_app = _read_json(APP_DIR / app_config)
        self.cfg_app = cfg_app
        mdir = APP_DIR / cfg_app["model_dir"]
        self.cfg = _read_json(mdir / "config.json")
        self.thresholds = _read_json(mdir / "concept_thresholds.json")
        self.kb = _read_json(APP_DIR / cfg_app["knowledge_file"])
        self.classes = self.cfg["classes"]
        self.concepts = self.cfg["concepts"]
        self.absence = self.cfg.get("absence_concept", "C16")
        self.uncertain_threshold = float(cfg_app["uncertain_threshold"])
        dev = device if device != "auto" else cfg_app.get("device", "auto")
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        t = time.perf_counter()
        self.model = build_model(self.cfg, "cbm", pretrained=False)  # khong tai trong so ImageNet: chay offline
        ck = torch.load(mdir / "best.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.load_seconds = time.perf_counter() - t
        size = self.cfg["image_size"]
        self.tf = T.Compose([PadSquare(), T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                             T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # Heatmap chi dung attention TANG CAO (stride 32): da chot phuong an A ngay 2026-11-09 sau khi
        # chan doan thay attention tang thap hay trai deu ra nen. Lay trong so attention qua hook, khong sua models.py.
        self._attn = None
        self.model.cross.register_forward_hook(self._grab_attention)
        with torch.no_grad():  # chay nong 1 lan de lan du doan dau tien khong bi cham
            self.model(torch.zeros(1, 3, size, size, device=self.device), return_attention=True)

    def _grab_attention(self, module, inputs, output):
        self._attn = output[1]  # (B, 14, so token tang thap + tang cao), trung binh qua cac head

    def _high_attention(self, heat_low_shape):
        """Tach phan attention cua tang cao tu trong so cross-attention -> (14, h_hi, w_hi)."""
        h_lo, w_lo = heat_low_shape
        a = self._attn[0].float()
        n_lo = h_lo * w_lo
        n_hi = a.shape[-1] - n_lo
        h_hi, w_hi = (h_lo + 1) // 2, (w_lo + 1) // 2
        if h_hi * w_hi != n_hi:
            raise RuntimeError(f"Khong tach duoc attention tang cao: {n_hi} token, luoi {h_hi}x{w_hi}")
        return a[:, n_lo:].reshape(-1, h_hi, w_hi).cpu().numpy()

    # ------------------------------------------------------------------ chinh
    @torch.no_grad()
    def predict(self, image):
        """image: duong dan hoac PIL.Image. Tra ve dict ket qua (xem README)."""
        t0 = time.perf_counter()
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        im = ImageOps.exif_transpose(image).convert("RGB")
        x = self.tf(im).unsqueeze(0).to(self.device)
        out = self.model(x, return_attention=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_model = time.perf_counter() - t0

        prob = out["class_logits"].float().softmax(1)[0].cpu().numpy()
        cprob = torch.sigmoid(out["concept_logits"].float())[0].cpu().numpy()
        heat = self._high_attention(out["heatmap"].shape[-2:])  # (14, h_hi, w_hi) - chi attention tang cao

        k = int(prob.argmax())
        pred = self.classes[k]
        uncertain = bool(prob[k] < self.uncertain_threshold)
        cp = {c: float(cprob[i]) for i, c in enumerate(self.concepts)}
        active = [c for c in self.concepts if cp[c] >= self.thresholds[c]]

        signs = [{"ma": c, "ten": self.kb["concepts"][c]["ten"], "do_tin_cay": cp[c]}
                 for c in sorted(active, key=lambda c: -cp[c])]
        # "Dau hieu co the co": xac suat >= 0.5 nhung chua dat nguong chon tren val. Chi de hien thi,
        # KHONG dung cho ket luan, ontology hay heatmap (chot ngay 2026-11-09).
        maybe = [c for c in self.concepts if c not in active and cp[c] >= MAYBE_MIN]
        maybe_signs = [{"ma": c, "ten": self.kb["concepts"][c]["ten"], "do_tin_cay": cp[c]}
                       for c in sorted(maybe, key=lambda c: -cp[c])]
        onto = self.check_ontology(pred, active, uncertain)
        heat_img, heat_note = self.make_heatmap(im, heat, cp, active)

        return {
            "ket_luan": "Không chắc chắn" if uncertain else self.kb["classes"][pred],
            "khong_chac_chan": uncertain,
            "lop_du_doan": pred,
            "xac_suat_benh": {self.kb["classes"][c]: float(prob[i]) for i, c in enumerate(self.classes)},  # khong hien tren app
            "xac_suat_concept": cp,
            "dau_hieu": signs,
            "dau_hieu_co_the": maybe_signs,
            "ontology": onto,
            "anh_hien_thi": self._display(im),
            "heatmap": heat_img,
            "ban_do_chu_y": self._last_map,  # mang HxW 0..1 cung kich thuoc anh hien thi (None neu khong ve); web tu pha mau
            "ghi_chu_heatmap": heat_note,
            "thoi_gian_mo_hinh_ms": 1000 * t_model,
            "thoi_gian_tong_ms": 1000 * (time.perf_counter() - t0),
            "thiet_bi": self.device.type,
        }

    # -------------------------------------------------------------- ontology
    def check_ontology(self, pred, active, uncertain):
        """Kiem tra cheo voi to hop P01-P04 (va rang buoc la khoe C16). Chi canh bao, KHONG sua ket luan.

        Trang thai:
          phu_hop  : du cac concept bat buoc DO DUOC cua to hop ung voi benh du doan
          mot_phan : khop k/n concept bat buoc (ontology: thieu to hop khong loai tru benh) -> chi thong tin
          canh_bao : mau thuan voi rang buoc la khoe C16 (du doan benh nhung C16 bat, hoac du doan la khoe nhung co ton thuong)
          khong_ap_dung : khi ket luan 'Khong chac chan'
        """
        lesion = [c for c in active if c != self.absence]
        if uncertain:
            return {"trang_thai": "khong_ap_dung", "thong_diep": "Không kiểm tra vì mô hình không chắc chắn.", "to_hop": None}
        if pred == "05_la_khoe":
            if self.absence in active and not lesion:
                return {"trang_thai": "phu_hop", "to_hop": "C16",
                        "thong_diep": "Phù hợp: không thấy tổn thương rõ trên phiến lá (C16), không có dấu hiệu bệnh nào."}
            names = ", ".join(lesion) if lesion else "C16 không bật"
            return {"trang_thai": "canh_bao", "to_hop": "C16",
                    "thong_diep": f"Cảnh báo: kết luận Lá khỏe nhưng phát hiện dấu hiệu tổn thương ({names})."}
        if self.absence in active:
            return {"trang_thai": "canh_bao", "to_hop": "C16",
                    "thong_diep": "Cảnh báo: kết luận có bệnh nhưng mô hình cũng thấy 'không có tổn thương rõ' (C16)."}
        p = next(t for t in self.kb["to_hop"] if t["benh"] == pred)
        need = p["bat_buoc_do_duoc"]
        hit = [c for c in need if c in active]
        miss = [c for c in need if c not in active]
        opt = [c for c in p["tuy_chon"] if c in active]
        res = {"to_hop": p["ma"], "ten_benh": p["ten_benh"], "trang": p["trang"], "khop": hit, "thieu": miss,
               "tuy_chon_co": opt, "chua_do": p["bat_buoc_chua_do"]}
        extra = ""  # nguoi dung yeu cau bo cau "(C.. chua do duoc bang mo hinh)" khoi thong diep (2026-11-09)
        if not miss:
            res.update(trang_thai="phu_hop",
                       thong_diep=f"Phù hợp tổ hợp {p['ma']} ({p['ten_benh']}): có đủ {', '.join(hit)}{extra}.")
        else:
            res.update(trang_thai="mot_phan",
                       thong_diep=f"Phù hợp một phần tổ hợp {p['ma']} ({p['ten_benh']}): khớp {len(hit)}/{len(need)}, "
                                  f"chưa thấy {', '.join(miss)}{extra}. Theo ontology, thiếu tổ hợp không loại trừ bệnh.")
        return res

    # --------------------------------------------------------------- heatmap
    def _display(self, im):
        w, h = im.size
        s = min(1.0, DISPLAY_MAX_SIDE / max(w, h))
        return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR) if s < 1 else im.copy()

    def make_heatmap(self, im, heat, cp, active):
        """Heatmap tong = trung binh attention TANG CAO cua cac concept ton thuong dang bat, trong so = xac suat concept.
        Neu khong co concept ton thuong nao (vd la khoe) thi khong ve heatmap."""
        lesion = [c for c in active if c != self.absence]
        disp = self._display(im)
        self._last_map = None
        if not lesion:
            return disp, "Không có dấu hiệu tổn thương nào được phát hiện nên không vẽ heatmap."
        idx = [self.concepts.index(c) for c in lesion]
        wts = np.array([cp[c] for c in lesion], dtype=np.float32)
        m = (heat[idx] * wts[:, None, None]).sum(0) / wts.sum()
        # anh vao mo hinh la anh da dem vuong -> phong heatmap len hinh vuong roi cat bo phan vien
        w, h = disp.size
        s = max(w, h)
        mt = F.interpolate(torch.from_numpy(m)[None, None], size=(s, s), mode="bilinear", align_corners=False)[0, 0].numpy()
        top, left = (s - h) // 2, (s - w) // 2
        mt = mt[top:top + h, left:left + w]
        mt = (mt - mt.min()) / (mt.max() - mt.min() + 1e-8)
        self._last_map = mt
        a = float(self.cfg_app.get("heatmap_alpha", 0.6))
        base = np.asarray(disp, dtype=np.float32) / 255.0
        return Image.fromarray(blend(base, mt, a)), "Vùng đỏ/vàng là nơi mô hình chú ý nhiều khi nhận các dấu hiệu."
