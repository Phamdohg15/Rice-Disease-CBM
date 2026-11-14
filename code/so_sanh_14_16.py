"""So sanh CBM 14 concept (runs/) voi CBM 16 concept (runs_16c/, them C08 + C13).
Chi doc ket qua da luu (metrics.json, *_concept_metrics.csv, *_classification_report.csv, *_predictions.csv),
khong train, khong can GPU. Chay bat cu luc nao: python so_sanh_14_16.py

Ket qua: so_sanh/SoSanh_14_vs_16_concept.xlsx  (bang so lieu + bieu do Excel)
         so_sanh/so_sanh_14_vs_16.png         (hinh co thanh sai so, dung cho luan van)

Luu y cach so sanh cong bang:
- Concept AUC / F1 trung binh duoc tinh lai CHI tren 14 concept chung cua hai ban.
  (Neu lay trung binh 16 concept thi C08 khong co anh duong tren val/test -> AUC khong tinh duoc,
  F1 = 0, keo trung binh xuong mot cach gia tao.)
- Ghep cap theo (backbone, seed): cung backbone, cung seed, cung tham so, chi khac so concept.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
ORDER = ["resnet18", "densenet121", "efficientnet_b0", "convnext_tiny", "mobilenet_v2"]
NAME = {"resnet18": "ResNet18", "densenet121": "DenseNet121", "efficientnet_b0": "EfficientNet-B0",
        "convnext_tiny": "ConvNeXt-Tiny", "mobilenet_v2": "MobileNetV2"}
SEEDS = [42, 43, 44]
C14 = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C09", "C10", "C11", "C12", "C14", "C15", "C16"]
EXTRA = ["C08", "C13"]
C16_ALL = [f"C{i:02d}" for i in range(1, 17)]
CONCEPT_NAME = {
    "C01": "Vết hình mắt én, hai đầu hẹp, giữa phình", "C02": "Phần giữa vết bệnh màu xám tro",
    "C03": "Viền vết bệnh màu nâu", "C04": "Vùng tổn thương kéo dài theo hướng gân lá",
    "C05": "Quầng vàng phía ngoài viền nâu của vết", "C06": "Đốm hình tròn hoặc bầu dục",
    "C07": "Phần chính của đốm trên lá màu nâu lợt", "C08": "Vòng đồng tâm bên trong vết đốm",
    "C09": "Tổn thương tiếp giáp chóp hoặc mép lá", "C10": "Vùng/vết tổn thương màu xám – xám trắng",
    "C11": "Viền ngoài vùng tổn thương có hình gợn sóng", "C12": "Nhiều vết bệnh liên kết với nhau",
    "C13": "Tổn thương trên bẹ lá", "C14": "Vết tổn thương có hình dạng không đều",
    "C15": "Vết loang lỗ, vằn vện như da hổ", "C16": "Không thấy tổn thương rõ trên phần phiến lá quan sát"}
CLASS_VN = {"01_dao_on": "Đạo ôn", "02_dom_nau": "Đốm nâu", "03_bac_la": "Bạc lá", "04_kho_van": "Khô vằn",
            "05_la_khoe": "Lá khỏe"}

# mau (bang mau tham chieu, da kiem tra phan biet duoc voi nguoi mu mau): xanh = 14, cam = 16, xam = baseline
COL14, COL16, COLBASE = "2A78D6", "EB6834", "B5B3AA"
HEAD_FILL = PatternFill("solid", fgColor="E8EEF7")
SUB_FILL = PatternFill("solid", fgColor="F4F4F1")
GOOD = Font(color="006300", bold=True)
BAD = Font(color="B42318", bold=True)
THIN = Side(style="thin", color="D0CFC8")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
NUM = "0.0000"


# ---------------------------------------------------------------- doc ket qua
def read_run(folder):
    m = folder / "metrics.json"
    if not m.exists():
        return None
    r = json.loads(m.read_text(encoding="utf-8"))
    r["_dir"] = folder
    return r


def concept_metrics(folder, split):
    p = folder / f"{split}_concept_metrics.csv"
    return pd.read_csv(p).set_index("concept") if p.exists() else None


def class_f1(folder, split="test"):
    p = folder / f"{split}_classification_report.csv"
    if not p.exists():
        return {}
    t = pd.read_csv(p, index_col=0)
    return {c: float(t.loc[c, "f1-score"]) for c in CLASS_VN if c in t.index}


def collect(runs14, runs16):
    rows, per_concept, per_class, extra = [], [], [], []
    for bb in ORDER:
        for s in SEEDS:
            rec = {"backbone": bb, "seed": s}
            base = read_run(runs14 / f"{bb}_baseline_seed{s}")
            if base:
                rec["base_test_f1"], rec["base_test_acc"] = base["test_f1"], base["test_acc"]
            for tag, root in (("14", runs14), ("16", runs16)):
                r = read_run(root / f"{bb}_cbm_seed{s}")
                if r is None:
                    continue
                d = r["_dir"]
                for k in ("test_f1", "test_acc", "val_f1", "val_acc", "epochs_run", "best_epoch", "parameters"):
                    rec[f"{k}_{tag}"] = r.get(k)
                rec[f"train_min_{tag}"] = r.get("training_seconds", np.nan) / 60
                for split in ("val", "test"):
                    ct = concept_metrics(d, split)
                    if ct is None:
                        continue
                    common = ct.loc[[c for c in C14 if c in ct.index]]
                    rec[f"{split}_cauc14_{tag}"] = float(common["auc"].mean())
                    rec[f"{split}_cf1_14_{tag}"] = float(common["f1"].mean())
                    if split == "test":
                        for c, row in ct.iterrows():
                            per_concept.append({"backbone": bb, "seed": s, "set": tag, "concept": c,
                                                "auc": row["auc"], "f1": row["f1"]})
                    if tag == "16":
                        pred = pd.read_csv(d / f"{split}_predictions.csv")
                        th = json.loads((d / "concept_thresholds.json").read_text(encoding="utf-8"))
                        for c in EXTRA:
                            if c not in ct.index:
                                continue
                            row = ct.loc[c]
                            extra.append({"backbone": bb, "seed": s, "split": split, "concept": c,
                                          "n_labeled": int(row["n_labeled"]), "n1": int(row["n1"]),
                                          "n0": int(row["n_labeled"] - row["n1"]), "threshold": th.get(c),
                                          "auc": row["auc"], "f1": row["f1"],
                                          "n_pred_1": int((pred[f"cp_{c}"] >= th.get(c, 0.5)).sum()),
                                          "n_images": len(pred)})
                for cls, f in class_f1(d).items():
                    per_class.append({"backbone": bb, "seed": s, "set": tag, "class": cls, "f1": f})
            rows.append(rec)
    df = pd.DataFrame(rows)
    return df, pd.DataFrame(per_concept), pd.DataFrame(per_class), pd.DataFrame(extra)


def label_counts():
    common = json.loads((HERE / "configs" / "common.json").read_text(encoding="utf-8"))
    out = []
    for split, f in common["label_files"].items():
        p = Path(common["label_dir"]) / f
        if not p.exists():
            continue
        lab = pd.read_csv(p)
        for c in EXTRA + ["C15"]:
            out.append({"split": split, "concept": c, "n1": int((lab[c] == 1).sum()),
                        "n0": int((lab[c] == 0).sum()), "n9": int((lab[c] == 9).sum()), "n": len(lab)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- tien ich Excel
def header(ws, row, values, col=1, fill=HEAD_FILL):
    for j, v in enumerate(values):
        c = ws.cell(row, col + j, v)
        c.font = Font(bold=True)
        c.fill = fill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BOX


def put(ws, row, col, v, fmt=None, delta=False, bold=False):
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        v = None
    if isinstance(v, np.generic):
        v = v.item()
    c = ws.cell(row, col, v)
    c.border = BOX
    if fmt and isinstance(v, (int, float)):
        c.number_format = fmt
    if delta and isinstance(v, (int, float)):
        c.number_format = "+0.0000;-0.0000;0.0000"
        if abs(v) >= 0.00005:
            c.font = GOOD if v > 0 else BAD
    elif bold:
        c.font = Font(bold=True)
    return c


def widths(ws, w):
    for i, x in enumerate(w, 1):
        ws.column_dimensions[get_column_letter(i)].width = x


def bar_chart(ws, title, ytitle, cats, data_cols, header_row, first, last, anchor, colors, ymin=None, ymax=None):
    ch = BarChart()
    ch.type = "col"
    ch.grouping = "clustered"
    ch.title = title
    ch.y_axis.title = ytitle
    ch.y_axis.majorGridlines = None
    ch.height, ch.width = 8.5, 18
    if ymin is not None:
        ch.y_axis.scaling.min = ymin
    if ymax is not None:
        ch.y_axis.scaling.max = ymax
    ch.y_axis.number_format = "0.00"
    ch.y_axis.delete = False
    ch.x_axis.delete = False
    for col, color in zip(data_cols, colors):
        ref = Reference(ws, min_col=col, min_row=header_row, max_row=last)
        ch.add_data(ref, titles_from_data=True)
        ch.series[-1].graphicalProperties.solidFill = color
        ch.series[-1].graphicalProperties.line.solidFill = "FFFFFF"
    ch.set_categories(Reference(ws, min_col=cats, min_row=first, max_row=last))
    ch.gapWidth = 60
    ch.x_axis.tickLblPos = "low"  # nhan backbone nam duoi truc khi co gia tri am
    ch.legend.position = "b"
    ws.add_chart(ch, anchor)


def ms(x):
    x = pd.Series(x).dropna()
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else np.nan, len(x)) if len(x) else (np.nan, np.nan, 0)


# ---------------------------------------------------------------- hinh cho luan van
def make_png(df, out):
    bbs = [b for b in ORDER if b in set(df.backbone)]
    x = np.arange(len(bbs))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    plt.rcParams["font.size"] = 10
    specs = [("base_test_f1", "Baseline (không concept)", "#" + COLBASE), ("test_f1_14", "CBM 14 concept", "#" + COL14),
             ("test_f1_16", "CBM 16 concept (+C08, C13)", "#" + COL16)]
    w = 0.26
    for k, (col, lab, color) in enumerate(specs):
        m = [ms(df.loc[df.backbone == b, col])[0] if col in df else np.nan for b in bbs]
        s = [ms(df.loc[df.backbone == b, col])[1] if col in df else np.nan for b in bbs]
        axs[0].bar(x + (k - 1) * w, m, w * 0.92, yerr=s, capsize=3, color=color, label=lab, edgecolor="white", linewidth=1.5)
    axs[0].set_title("(a) Macro-F1 chẩn đoán bệnh trên tập test")
    axs[0].set_ylabel("Macro-F1 (trung bình ± độ lệch chuẩn, 3 seed)")
    axs[0].set_ylim(0.70, 1.0)
    specs_auc = [("test_cauc14_14", specs[1][1], specs[1][2]), ("test_cauc14_16", specs[2][1], specs[2][2])]
    for k, (col, lab, color) in enumerate(specs_auc):
        m = [ms(df.loc[df.backbone == b, col])[0] if col in df else np.nan for b in bbs]
        s = [ms(df.loc[df.backbone == b, col])[1] if col in df else np.nan for b in bbs]
        axs[1].bar(x + (k - 0.5) * 0.36, m, 0.33, yerr=s, capsize=3, color=color, label=lab, edgecolor="white", linewidth=1.5)
    axs[1].set_title("(b) AUC concept trung bình trên 14 concept chung (test)")
    axs[1].set_ylabel("AUC (trung bình ± độ lệch chuẩn, 3 seed)")
    axs[1].set_ylim(0.80, 1.0)
    for ax in axs:
        ax.set_xticks(x, [NAME[b] for b in bbs], rotation=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.33), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    runs14 = HERE / "runs"
    runs16 = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs_16c"
    df, pc, pcls, extra = collect(runs14, runs16)
    have16 = df["test_f1_16"].notna().sum() if "test_f1_16" in df else 0
    if have16 == 0:
        print("Chua co luot 16 concept nao xong trong", runs16)
        return 1
    pairs = df.dropna(subset=["test_f1_14", "test_f1_16"]).copy()
    for k in ("test_f1", "test_acc", "val_f1", "test_cauc14", "test_cf1_14"):
        pairs[f"d_{k}"] = pairs[f"{k}_16"] - pairs[f"{k}_14"]
    out_dir = HERE / "so_sanh"
    out_dir.mkdir(exist_ok=True)
    png = out_dir / "so_sanh_14_vs_16.png"
    make_png(df, png)

    wb = Workbook()

    # ===== Sheet 1: Doc truoc
    ws = wb.active
    ws.title = "Doc_truoc"
    n_pairs = len(pairs)
    d = pairs["d_test_f1"]
    try:
        from scipy.stats import wilcoxon
        p_w = float(wilcoxon(d).pvalue) if n_pairs >= 6 and (d != 0).any() else np.nan
    except Exception:
        p_w = np.nan
    lines = [
        ("SO SÁNH CBM 14 CONCEPT VÀ CBM 16 CONCEPT (ABLATION C08, C13)", True),
        ("", False),
        ("Mục đích", True),
        ("Kiểm tra việc loại C08 (vòng đồng tâm) và C13 (tổn thương trên bẹ lá) khỏi bottleneck có làm thay đổi kết quả chẩn đoán "
         "và chất lượng dự đoán các concept còn lại hay không.", False),
        ("", False),
        ("Thiết lập", True),
        ("• CBM 14 concept: các lượt đã chạy trong thư mục runs/ (C01–C07, C09–C12, C14–C16).", False),
        ("• CBM 16 concept: thư mục runs_16c/, thêm C08 và C13 vào bottleneck.", False),
        ("• Hai bản giống hệt nhau về backbone, siêu tham số, dữ liệu, cách tiền xử lý, seed (42, 43, 44); chỉ khác số concept. "
         "pos_weight của C08 tính như mọi concept (544/7 → bị chặn ở 10).", False),
        ("• Mô hình tốt nhất chọn theo val macro-F1 (rồi val CE), ngưỡng concept chọn trên val, áp dụng cho test.", False),
        ("• Baseline (không concept) chỉ để tham chiếu.", False),
        ("", False),
        ("Cách đọc", True),
        ("• Δ = (16 concept) − (14 concept). Màu xanh: bản 16 cao hơn; màu đỏ: bản 16 thấp hơn.", False),
        ("• Mỗi ô là trung bình qua 3 seed; std là độ lệch chuẩn giữa 3 seed. Nếu |Δ| nhỏ hơn std thì chênh lệch nằm trong mức dao động do seed.", False),
        ("• AUC/F1 concept trung bình chỉ tính trên 14 concept chung để hai bản so được với nhau.", False),
        ("", False),
        ("Giới hạn cần nêu khi báo cáo", True),
        ("• C08 không có ảnh nhãn 1 nào trong val và test → không tính được AUC/F1 cho C08; chỉ đếm được số ảnh mô hình báo nhầm C08 (sheet C08_C13).", False),
        ("• C13 chỉ có 3 ảnh có nhãn (đều nhãn 1) trong val và 8 ảnh trong test → AUC của C13 (nếu có) không đáng tin.", False),
        ("• Vì vậy thí nghiệm này trả lời được: thêm C08, C13 có ảnh hưởng đến chẩn đoán bệnh và 14 concept còn lại không; "
         "KHÔNG chứng minh được mô hình học được C08, C13.", False),
        ("", False),
        ("Số liệu tóm tắt (tính tự động)", True),
        (f"• Số cặp (backbone, seed) đã có cả hai bản: {n_pairs} / 15.", False),
        (f"• Δ test macro-F1 trung bình qua các cặp: {d.mean():+.4f} (std {d.std(ddof=1):.4f}); "
         f"số cặp bản 16 cao hơn: {(d > 0).sum()}, thấp hơn: {(d < 0).sum()}, bằng: {(d == 0).sum()}.", False),
        (f"• Kiểm định Wilcoxon cặp đôi trên Δ test macro-F1: p = {p_w:.3f}" if not np.isnan(p_w)
         else "• Kiểm định Wilcoxon: chưa đủ cặp (cần ≥ 6) hoặc mọi Δ bằng 0.", False),
        (f"• Δ AUC concept (14 concept chung, test) trung bình: {pairs['d_test_cauc14'].mean():+.4f}.", False),
        ("", False),
        ("Các sheet", True),
        ("Tong_hop: bảng chính theo backbone + biểu đồ | Tung_luot: từng lượt, từng seed | AUC_tung_concept: AUC từng concept | "
         "F1_theo_lop: F1 từng bệnh | C08_C13: số liệu nhãn và dự đoán của hai concept bị loại | Bieu_do: hình cho luận văn", False),
    ]
    for i, (t, b) in enumerate(lines, 1):
        c = ws.cell(i, 1, t)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if b:
            c.font = Font(bold=True, size=13 if i == 1 else 11)
    ws.column_dimensions["A"].width = 140

    # ===== Sheet 2: Tong hop
    ws = wb.create_sheet("Tong_hop")
    metrics = [("Test macro-F1", "test_f1"), ("Test accuracy", "test_acc"), ("Val macro-F1", "val_f1"),
               ("AUC concept (14 chung, test)", "test_cauc14"), ("F1 concept (14 chung, test)", "test_cf1_14")]
    ws.cell(1, 1, "Bảng tổng hợp: trung bình (std) qua 3 seed. Δ = 16 − 14.").font = Font(bold=True, size=12)
    r = 3
    head = ["Backbone", "Số seed", "Baseline test F1"]
    for lab, _ in metrics:
        head += [f"{lab}\n14 concept", "std", f"{lab}\n16 concept", "std", "Δ (16−14)"]
    header(ws, r, head)
    ws.row_dimensions[r].height = 48
    first = r + 1
    summary = []
    for bb in ORDER:
        g = df[df.backbone == bb]
        if g.get("test_f1_16") is None or g["test_f1_16"].notna().sum() == 0:
            continue
        r += 1
        put(ws, r, 1, NAME[bb], bold=True)
        put(ws, r, 2, int(g["test_f1_16"].notna().sum()))
        put(ws, r, 3, ms(g.get("base_test_f1", pd.Series(dtype=float)))[0], NUM)
        col = 4
        srow = {"Backbone": NAME[bb]}
        for lab, k in metrics:
            m14, s14, _ = ms(g[f"{k}_14"])
            m16, s16, _ = ms(g[f"{k}_16"])
            put(ws, r, col, m14, NUM); put(ws, r, col + 1, s14, NUM)
            put(ws, r, col + 2, m16, NUM); put(ws, r, col + 3, s16, NUM)
            put(ws, r, col + 4, m16 - m14, delta=True)
            srow[k] = (m14, m16)
            col += 5
        summary.append(srow)
    last = r
    r += 1
    put(ws, r, 1, "Trung bình 5 backbone", bold=True)
    for cidx in range(3, 4 + 5 * len(metrics)):
        hdr = ws.cell(3, cidx).value
        if hdr == "std":
            put(ws, r, cidx, None)
            continue
        L = get_column_letter(cidx)
        c = ws.cell(r, cidx, f"=AVERAGE({L}{first}:{L}{last})")
        c.border = BOX
        c.font = Font(bold=True)
        c.number_format = "+0.0000;-0.0000;0.0000" if str(hdr).startswith("Δ") else NUM
    widths(ws, [18, 7, 11] + [11, 7, 11, 7, 10] * len(metrics))
    ws.freeze_panes = "B4"

    # bang gon cho bieu do
    cr = r + 3
    ws.cell(cr - 1, 1, "Dữ liệu cho biểu đồ (trung bình 3 seed)").font = Font(bold=True)
    header(ws, cr, ["Backbone", "Baseline", "CBM 14 concept", "CBM 16 concept", "", "Backbone", "AUC 14 concept", "AUC 16 concept",
                    "", "Backbone", "Δ test F1 (16−14)"])
    for i, srow in enumerate(summary, 1):
        bb = [b for b in ORDER if NAME[b] == srow["Backbone"]][0]
        put(ws, cr + i, 1, srow["Backbone"])
        put(ws, cr + i, 2, ms(df.loc[df.backbone == bb, "base_test_f1"])[0] if "base_test_f1" in df else None, NUM)
        put(ws, cr + i, 3, srow["test_f1"][0], NUM); put(ws, cr + i, 4, srow["test_f1"][1], NUM)
        put(ws, cr + i, 6, srow["Backbone"])
        put(ws, cr + i, 7, srow["test_cauc14"][0], NUM); put(ws, cr + i, 8, srow["test_cauc14"][1], NUM)
        put(ws, cr + i, 10, srow["Backbone"])
        put(ws, cr + i, 11, srow["test_f1"][1] - srow["test_f1"][0], delta=True)
    cl = cr + len(summary)
    anchor_row = cl + 3
    bar_chart(ws, "Macro-F1 chẩn đoán trên tập test", "Macro-F1", 1, [2, 3, 4], cr, cr + 1, cl, f"A{anchor_row}",
              [COLBASE, COL14, COL16], ymin=0.7, ymax=1.0)
    bar_chart(ws, "AUC concept trung bình (14 concept chung, test)", "AUC", 6, [7, 8], cr, cr + 1, cl, f"L{anchor_row}",
              [COL14, COL16], ymin=0.8, ymax=1.0)
    bar_chart(ws, "Δ test macro-F1 (16 − 14); > 0: thêm C08, C13 cao hơn", "Δ macro-F1", 10, [11], cr, cr + 1, cl,
              f"A{anchor_row + 19}", [COL16])

    # ===== Sheet 3: Tung luot
    ws = wb.create_sheet("Tung_luot")
    cols = [("Backbone", None), ("Seed", None), ("Baseline test F1", "base_test_f1"),
            ("Test F1 14", "test_f1_14"), ("Test F1 16", "test_f1_16"), ("Δ test F1", "d_test_f1"),
            ("Test acc 14", "test_acc_14"), ("Test acc 16", "test_acc_16"), ("Δ test acc", "d_test_acc"),
            ("Val F1 14", "val_f1_14"), ("Val F1 16", "val_f1_16"), ("Δ val F1", "d_val_f1"),
            ("AUC concept 14 chung (test) - bản 14", "test_cauc14_14"), ("AUC concept 14 chung (test) - bản 16", "test_cauc14_16"),
            ("Δ AUC", "d_test_cauc14"), ("Epoch chạy 14", "epochs_run_14"), ("Epoch chạy 16", "epochs_run_16"),
            ("Best epoch 14", "best_epoch_14"), ("Best epoch 16", "best_epoch_16"),
            ("Phút train 14", "train_min_14"), ("Phút train 16", "train_min_16"),
            ("Tham số 14", "parameters_14"), ("Tham số 16", "parameters_16")]
    header(ws, 1, [c for c, _ in cols])
    ws.row_dimensions[1].height = 45
    full = df.merge(pairs[["backbone", "seed"] + [c for c in pairs if c.startswith("d_")]], on=["backbone", "seed"], how="left")
    for i, row in enumerate(full.itertuples(index=False), 2):
        row = row._asdict()
        for j, (_, k) in enumerate(cols, 1):
            if k is None:
                put(ws, i, j, NAME[row["backbone"]] if j == 1 else row["seed"])
            elif k.startswith("d_"):
                put(ws, i, j, row.get(k), delta=True)
            elif k.startswith(("epochs", "best", "param")):
                put(ws, i, j, row.get(k), "0")
            elif k.startswith("train_min"):
                put(ws, i, j, row.get(k), "0.0")
            else:
                put(ws, i, j, row.get(k), NUM)
    widths(ws, [16, 6] + [11] * (len(cols) - 2))
    ws.freeze_panes = "C2"

    # ===== Sheet 4: AUC tung concept
    ws = wb.create_sheet("AUC_tung_concept")
    ws.cell(1, 1, "AUC từng concept trên tập test (trung bình 3 seed). Ô trống = không tính được AUC "
                  "(concept không có trong bản đó, hoặc tập test thiếu nhãn 0 hoặc nhãn 1).").font = Font(bold=True)
    head = ["Mã", "Tên concept"]
    for bb in ORDER:
        head += [f"{NAME[bb]}\n14", f"{NAME[bb]}\n16", "Δ"]
    head += ["TB 5 backbone\n14", "TB 5 backbone\n16", "Δ TB"]
    header(ws, 3, head)
    ws.row_dimensions[3].height = 45
    piv = pc.groupby(["concept", "set", "backbone"])["auc"].mean() if len(pc) else pd.Series(dtype=float)
    chart_rows = []
    for i, c in enumerate(C16_ALL, 4):
        put(ws, i, 1, c, bold=True)
        put(ws, i, 2, CONCEPT_NAME[c] + ("  (chỉ có ở bản 16)" if c in EXTRA else ""))
        if c in EXTRA:
            for j in (1, 2):
                ws.cell(i, j).fill = PatternFill("solid", fgColor="FDEBDD")
        col = 3
        a14, a16 = [], []
        for bb in ORDER:
            v14 = piv.get((c, "14", bb), np.nan)
            v16 = piv.get((c, "16", bb), np.nan)
            a14.append(v14); a16.append(v16)
            put(ws, i, col, v14, "0.000"); put(ws, i, col + 1, v16, "0.000")
            put(ws, i, col + 2, v16 - v14 if not (np.isnan(v14) or np.isnan(v16)) else None, delta=True)
            col += 3
        m14, m16 = np.nanmean(a14) if np.isfinite(a14).any() else np.nan, np.nanmean(a16) if np.isfinite(a16).any() else np.nan
        put(ws, i, col, m14, "0.000"); put(ws, i, col + 1, m16, "0.000")
        put(ws, i, col + 2, m16 - m14 if not (np.isnan(m14) or np.isnan(m16)) else None, delta=True)
        if c in C14:
            chart_rows.append((c, m14, m16))
    widths(ws, [6, 44] + [9, 9, 9] * (len(ORDER) + 1))
    ws.freeze_panes = "C4"
    cr = 4 + len(C16_ALL) + 2
    header(ws, cr, ["Concept", "CBM 14 concept", "CBM 16 concept"])
    for i, (c, m14, m16) in enumerate(chart_rows, 1):
        put(ws, cr + i, 1, c); put(ws, cr + i, 2, m14, "0.000"); put(ws, cr + i, 3, m16, "0.000")
    bar_chart(ws, "AUC từng concept (TB 5 backbone × 3 seed, test) – 14 concept chung", "AUC", 1, [2, 3], cr, cr + 1,
              cr + len(chart_rows), f"E{cr}", [COL14, COL16], ymin=0.7, ymax=1.0)

    # ===== Sheet 5: F1 theo lop
    ws = wb.create_sheet("F1_theo_lop")
    ws.cell(1, 1, "F1 từng bệnh trên tập test (trung bình 3 seed). C08 gắn với Đốm nâu (P02), C13 gắn với Khô vằn (P04).").font = Font(bold=True)
    head = ["Lớp"]
    for bb in ORDER:
        head += [f"{NAME[bb]}\n14", f"{NAME[bb]}\n16", "Δ"]
    head += ["TB 5 backbone\n14", "TB 5 backbone\n16", "Δ TB"]
    header(ws, 3, head)
    ws.row_dimensions[3].height = 45
    pv = pcls.groupby(["class", "set", "backbone"])["f1"].mean() if len(pcls) else pd.Series(dtype=float)
    crow = []
    for i, cls in enumerate(CLASS_VN, 4):
        put(ws, i, 1, CLASS_VN[cls], bold=True)
        col, a14, a16 = 2, [], []
        for bb in ORDER:
            v14, v16 = pv.get((cls, "14", bb), np.nan), pv.get((cls, "16", bb), np.nan)
            a14.append(v14); a16.append(v16)
            put(ws, i, col, v14, NUM); put(ws, i, col + 1, v16, NUM)
            put(ws, i, col + 2, v16 - v14 if not (np.isnan(v14) or np.isnan(v16)) else None, delta=True)
            col += 3
        m14, m16 = np.nanmean(a14), np.nanmean(a16)
        put(ws, i, col, m14, NUM); put(ws, i, col + 1, m16, NUM); put(ws, i, col + 2, m16 - m14, delta=True)
        crow.append((CLASS_VN[cls], m14, m16))
    widths(ws, [12] + [9, 9, 9] * (len(ORDER) + 1))
    cr = 4 + len(CLASS_VN) + 2
    header(ws, cr, ["Lớp", "CBM 14 concept", "CBM 16 concept"])
    for i, (c, m14, m16) in enumerate(crow, 1):
        put(ws, cr + i, 1, c); put(ws, cr + i, 2, m14, NUM); put(ws, cr + i, 3, m16, NUM)
    bar_chart(ws, "F1 từng bệnh trên test (TB 5 backbone × 3 seed)", "F1", 1, [2, 3], cr, cr + 1, cr + len(crow), f"E{cr}",
              [COL14, COL16], ymin=0.6, ymax=1.0)

    # ===== Sheet 6: C08_C13
    ws = wb.create_sheet("C08_C13")
    ws.cell(1, 1, "A. Số ảnh theo nhãn trong file nhãn V1.8 (1 = có, 0 = không, 9 = không xác định). C15 là concept ít ảnh nhất còn giữ, để đối chiếu.").font = Font(bold=True)
    header(ws, 2, ["Tập", "Concept", "Tên", "n1", "n0", "n9", "Tổng ảnh", "Đạt ngưỡng ≥80 (n1 và n0, chỉ xét train)"])
    lc = label_counts()
    for i, row in enumerate(lc.itertuples(index=False), 3):
        put(ws, i, 1, row.split); put(ws, i, 2, row.concept, bold=True); put(ws, i, 3, CONCEPT_NAME[row.concept])
        put(ws, i, 4, row.n1); put(ws, i, 5, row.n0); put(ws, i, 6, row.n9); put(ws, i, 7, row.n)
        put(ws, i, 8, ("Đạt" if row.n1 >= 80 and row.n0 >= 80 else "Không đạt") if row.split == "train" else "")
    r = 3 + len(lc) + 2
    ws.cell(r, 1, "B. Bản 16 concept: mô hình dự đoán C08, C13 như thế nào. "
                  "\"Số ảnh báo có\" = số ảnh có xác suất ≥ ngưỡng (ngưỡng chọn trên val; nếu val thiếu nhãn thì ngưỡng = 0.5).").font = Font(bold=True)
    header(ws, r + 1, ["Backbone", "Seed", "Tập", "Concept", "Ảnh có nhãn 0/1", "n1", "n0", "Ngưỡng", "AUC", "F1",
                       "Số ảnh báo có", "Tổng ảnh"])
    for i, row in enumerate(extra.sort_values(["concept", "split", "backbone", "seed"],
                                              key=lambda s: s.map(ORDER.index) if s.name == "backbone" else s).itertuples(index=False),
                            r + 2):
        put(ws, i, 1, NAME[row.backbone]); put(ws, i, 2, row.seed); put(ws, i, 3, row.split); put(ws, i, 4, row.concept, bold=True)
        put(ws, i, 5, row.n_labeled); put(ws, i, 6, row.n1); put(ws, i, 7, row.n0); put(ws, i, 8, row.threshold, "0.00")
        put(ws, i, 9, row.auc, "0.000"); put(ws, i, 10, row.f1 if row.n1 > 0 else None, "0.000")
        put(ws, i, 11, row.n_pred_1); put(ws, i, 12, row.n_images)
    widths(ws, [16, 8, 8, 9, 14, 7, 7, 9, 9, 9, 13, 10])

    # ===== Sheet 7: Bieu do
    ws = wb.create_sheet("Bieu_do")
    ws.cell(1, 1, "Hình cho luận văn (cũng lưu riêng: so_sanh/so_sanh_14_vs_16.png). Thanh sai số = độ lệch chuẩn qua 3 seed.").font = Font(bold=True)
    img = XLImage(str(png))
    img.width, img.height = img.width * 0.42, img.height * 0.42
    ws.add_image(img, "A3")

    xlsx = out_dir / "SoSanh_14_vs_16_concept.xlsx"
    try:
        wb.save(xlsx)
    except PermissionError:
        xlsx = out_dir / "SoSanh_14_vs_16_concept_moi.xlsx"
        wb.save(xlsx)
        print("File Excel cu dang mo, da luu ban moi:", xlsx)
    print(f"Da ghep {n_pairs} cap (backbone, seed). Delta test F1 TB = {d.mean():+.4f}")
    print("Da luu:", xlsx)
    print("Da luu:", png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
