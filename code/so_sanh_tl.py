"""So sanh cac chien luoc transfer learning. Chi doc ket qua da luu, khong train.
  A full_ft   : fine-tune toan bo tu ImageNet (runs/, cac luot da co)
  B two_stage : hai giai doan - dong bang backbone roi mo khoi stride-32 (runs_tl/)
  D frozen    : chi giai doan 1 - backbone dong bang hoan toan (runs_tl/)
  C scratch   : khong dung trong so ImageNet (runs_tl/)
Chay: python so_sanh_tl.py   -> so_sanh/SoSanh_TransferLearning.xlsx + so_sanh/so_sanh_transfer_learning.png"""
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
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
ORDER = ["resnet18", "densenet121", "efficientnet_b0", "convnext_tiny", "mobilenet_v2"]
NAME = {"resnet18": "ResNet18", "densenet121": "DenseNet121", "efficientnet_b0": "EfficientNet-B0",
        "convnext_tiny": "ConvNeXt-Tiny", "mobilenet_v2": "MobileNetV2"}
SEEDS = [42, 43, 44]
MODES = [("baseline", "Baseline"), ("cbm", "CBM")]
# (ma, ten thu muc, nhan, mau) - mau theo bang mau tham chieu da kiem tra mu mau (4 o dau, dung thu tu)
STRATS = [("A", "full_ft", "A. Fine-tune toàn bộ (đang dùng)", "2A78D6"),
          ("B", "two_stage", "B. Hai giai đoạn (theo notebook của thầy)", "EB6834"),
          ("D", "frozen", "D. Chỉ đóng băng backbone (GĐ1)", "1BAF7A"),
          ("C", "scratch", "C. Huấn luyện từ đầu", "EDA100")]
SHORT = {"A": "A. Fine-tune toàn bộ", "B": "B. Hai giai đoạn", "D": "D. Đóng băng backbone", "C": "C. Từ đầu"}
CLASS_VN = {"01_dao_on": "Đạo ôn", "02_dom_nau": "Đốm nâu", "03_bac_la": "Bạc lá", "04_kho_van": "Khô vằn",
            "05_la_khoe": "Lá khỏe"}

HEAD_FILL = PatternFill("solid", fgColor="E8EEF7")
GOOD = Font(color="006300", bold=True)
BAD = Font(color="B42318", bold=True)
THIN = Side(style="thin", color="D0CFC8")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
NUM = "0.0000"
DELTA = "+0.0000;-0.0000;0.0000"


# ---------------------------------------------------------------- doc ket qua
def run_dir(code, folder, bb, mode, seed, runs, runs_tl):
    if code == "A":
        return runs / f"{bb}_{mode}_seed{seed}"
    return runs_tl / f"{bb}_{mode}_{folder}_seed{seed}"


def collect(runs, runs_tl):
    rows, concepts, classes = [], [], []
    for bb in ORDER:
        for mode, _ in MODES:
            for s in SEEDS:
                for code, folder, _, _ in STRATS:
                    d = run_dir(code, folder, bb, mode, s, runs, runs_tl)
                    m = d / "metrics.json"
                    if not m.exists():
                        continue
                    r = json.loads(m.read_text(encoding="utf-8"))
                    rows.append({"backbone": bb, "mode": mode, "seed": s, "strategy": code,
                                 "test_f1": r.get("test_f1"), "test_acc": r.get("test_acc"), "val_f1": r.get("val_f1"),
                                 "test_cauc": r.get("test_concept_auc"), "test_cf1": r.get("test_concept_f1"),
                                 "epochs_run": r.get("epochs_run"), "best_epoch": r.get("best_epoch"),
                                 "train_min": r.get("training_seconds", np.nan) / 60,
                                 "trainable_params": r.get("trainable_params", r.get("parameters")),
                                 "parameters": r.get("parameters")})
                    cm = d / "test_concept_metrics.csv"
                    if mode == "cbm" and cm.exists():
                        t = pd.read_csv(cm)
                        for _, c in t.iterrows():
                            concepts.append({"backbone": bb, "seed": s, "strategy": code, "concept": c["concept"], "auc": c["auc"]})
                    cr = d / "test_classification_report.csv"
                    if cr.exists():
                        t = pd.read_csv(cr, index_col=0)
                        for cls in CLASS_VN:
                            if cls in t.index:
                                classes.append({"backbone": bb, "mode": mode, "seed": s, "strategy": code,
                                                "class": cls, "f1": float(t.loc[cls, "f1-score"])})
    return pd.DataFrame(rows), pd.DataFrame(concepts), pd.DataFrame(classes)


def ms(x):
    x = pd.Series(x, dtype=float).dropna()
    if not len(x):
        return np.nan, np.nan, 0
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else np.nan), len(x)


def paired(df, mode, a, b, col="test_f1"):
    """Chenh lech ghep cap (backbone, seed): b - a."""
    x = df[(df["mode"] == mode) & (df.strategy == a)].set_index(["backbone", "seed"])[col]
    y = df[(df["mode"] == mode) & (df.strategy == b)].set_index(["backbone", "seed"])[col]
    j = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna()
    return (j["b"] - j["a"])


def wilcoxon_p(d):
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue) if len(d) >= 6 and (d != 0).any() else np.nan
    except Exception:
        return np.nan


# ---------------------------------------------------------------- tien ich Excel
def header(ws, row, values, col=1):
    for j, v in enumerate(values):
        c = ws.cell(row, col + j, v)
        c.font = Font(bold=True)
        c.fill = HEAD_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BOX


def put(ws, row, col, v, fmt=None, delta=False, bold=False):
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        v = None
    if isinstance(v, np.generic):
        v = v.item()
    c = ws.cell(row, col, v)
    c.border = BOX
    if delta and isinstance(v, (int, float)):
        c.number_format = DELTA
        if abs(v) >= 0.00005:
            c.font = GOOD if v > 0 else BAD
    else:
        if fmt and isinstance(v, (int, float)):
            c.number_format = fmt
        if bold:
            c.font = Font(bold=True)
    return c


def widths(ws, w):
    for i, x in enumerate(w, 1):
        ws.column_dimensions[get_column_letter(i)].width = x


def bar_chart(ws, title, ytitle, cat_col, data_cols, header_row, first, last, anchor, colors, ymin=None, ymax=None):
    ch = BarChart()
    ch.type, ch.grouping = "col", "clustered"
    ch.title, ch.y_axis.title = title, ytitle
    ch.y_axis.majorGridlines = None
    ch.height, ch.width = 8.5, 20
    if ymin is not None:
        ch.y_axis.scaling.min = ymin
    if ymax is not None:
        ch.y_axis.scaling.max = ymax
    ch.y_axis.number_format = "0.00"
    ch.y_axis.delete = False
    ch.x_axis.delete = False
    ch.x_axis.tickLblPos = "low"
    for col, color in zip(data_cols, colors):
        ch.add_data(Reference(ws, min_col=col, min_row=header_row, max_row=last), titles_from_data=True)
        ch.series[-1].graphicalProperties.solidFill = color
        ch.series[-1].graphicalProperties.line.solidFill = "FFFFFF"
    ch.set_categories(Reference(ws, min_col=cat_col, min_row=first, max_row=last))
    ch.gapWidth = 60
    ch.legend.position = "b"
    ws.add_chart(ch, anchor)


def metric_block(ws, r, df, mode, col, title, present):
    """Bang: dong = backbone, cot = chien luoc (TB, std) + chenh lech so voi A. Tra ve (dong ke tiep, vi tri cot TB)."""
    ws.cell(r, 1, title).font = Font(bold=True, size=12)
    r += 1
    head = ["Backbone"]
    mean_cols = {}
    for code in present:
        mean_cols[code] = len(head) + 1
        head += [SHORT[code], "std", "n"]
    for code in present:
        if code != "A":
            head.append(f"{code} − A")
    header(ws, r, head)
    ws.row_dimensions[r].height = 42
    hr, first = r, r + 1
    for bb in ORDER:
        g = df[(df.backbone == bb) & (df["mode"] == mode)]
        if not len(g):
            continue
        r += 1
        put(ws, r, 1, NAME[bb], bold=True)
        c = 2
        means = {}
        for code in present:
            m, s, n = ms(g.loc[g.strategy == code, col])
            means[code] = m
            put(ws, r, c, m, NUM); put(ws, r, c + 1, s, NUM); put(ws, r, c + 2, n, "0")
            c += 3
        for code in present:
            if code != "A":
                put(ws, r, c, means[code] - means.get("A", np.nan) if not np.isnan(means.get("A", np.nan)) else None, delta=True)
                c += 1
    last = r
    r += 1
    put(ws, r, 1, "Trung bình 5 backbone", bold=True)
    for cidx in range(2, len(head) + 1):
        h = ws.cell(hr, cidx).value
        if h in ("std", "n"):
            put(ws, r, cidx, None)
            continue
        L = get_column_letter(cidx)
        c = ws.cell(r, cidx, f"=AVERAGE({L}{first}:{L}{last})")
        c.border, c.font = BOX, Font(bold=True)
        c.number_format = DELTA if "−" in str(h) else NUM
    return r + 3, hr, first, last, mean_cols


# ---------------------------------------------------------------- hinh cho luan van
def make_png(df, present, out):
    panels = [("baseline", "test_f1", "(a) Baseline: macro-F1 trên tập test", (0.4, 1.0)),
              ("cbm", "test_f1", "(b) CBM: macro-F1 trên tập test", (0.4, 1.0)),
              ("cbm", "test_cauc", "(c) CBM: AUC concept trung bình (test)", (0.5, 1.0))]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    x = np.arange(len(ORDER))
    k = len(present)
    w = 0.8 / k
    for ax, (mode, col, title, ylim) in zip(axs, panels):
        for i, code in enumerate(present):
            color = "#" + dict((s[0], s[3]) for s in STRATS)[code]
            m = [ms(df.loc[(df.backbone == b) & (df["mode"] == mode) & (df.strategy == code), col])[0] for b in ORDER]
            s = [ms(df.loc[(df.backbone == b) & (df["mode"] == mode) & (df.strategy == code), col])[1] for b in ORDER]
            ax.bar(x + (i - (k - 1) / 2) * w, m, w * 0.92, yerr=s, capsize=2.5, color=color, label=SHORT[code],
                   edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(*ylim)
        ax.set_xticks(x, [NAME[b] for b in ORDER], rotation=15, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
    axs[0].set_ylabel("Trung bình ± độ lệch chuẩn (3 seed)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=k, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    runs = HERE / "runs"
    runs_tl = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs_tl"
    if not runs_tl.is_absolute():
        runs_tl = HERE / runs_tl
    df, pc, pcls = collect(runs, runs_tl)
    present = [s[0] for s in STRATS if s[0] in set(df.strategy)] if len(df) else []
    if len(set(present) - {"A"}) == 0:
        print("Chua co luot transfer learning nao xong trong", runs_tl)
        return 1
    tl = json.loads((HERE / "configs" / "transfer.json").read_text(encoding="utf-8"))
    out_dir = HERE / "so_sanh"
    out_dir.mkdir(exist_ok=True)
    png = out_dir / "so_sanh_transfer_learning.png"
    make_png(df, present, png)
    wb = Workbook()

    # ===== Doc_truoc
    ws = wb.active
    ws.title = "Doc_truoc"
    L = [("SO SÁNH CÁC CHIẾN LƯỢC TRANSFER LEARNING (baseline và CBM, 5 backbone × 3 seed)", "h1"), ("", None),
         ("Bốn chiến lược", "h"),
         ("A. Fine-tune toàn bộ từ ImageNet (cách đang dùng, kết quả trong runs/): mọi tầng học ngay từ đầu; backbone lr 1e-4, "
          "phần CBM mới lr 1e-3; tối đa 60 epoch, dừng sớm sau 12 epoch không cải thiện; BatchNorm cập nhật theo batch.", None),
         (f"B. Hai giai đoạn (theo notebook Transfer_Learning của thầy): GĐ1 đóng băng toàn bộ backbone, chỉ học phần đầu mới "
          f"(baseline: lớp Linear cuối; CBM: chiếu 1×1, Transformer, truy vấn concept, đầu concept, MLP), lr {tl['stage1']['learning_rate']:g}, "
          f"tối đa {tl['stage1']['epochs']} epoch, dừng sớm {tl['stage1']['patience']}. GĐ2 bắt đầu từ mô hình tốt nhất của GĐ1, mở các khối "
          f"stride-32 của backbone, BatchNorm vẫn đóng băng, lr {tl['stage2']['learning_rate']:g} cho mọi tham số đang học, "
          f"tối đa {tl['stage2']['epochs']} epoch, dừng sớm {tl['stage2']['patience']}.", None),
         ("D. Chỉ đóng băng backbone: chính là mô hình tốt nhất sau GĐ1 của B (không cần chạy thêm). "
          "Cho biết đặc trưng ImageNet chưa fine-tune đã đủ đến đâu.", None),
         ("C. Huấn luyện từ đầu: khởi tạo ngẫu nhiên, không dùng trọng số ImageNet; mọi thiết lập khác giống hệt A "
          "(cố ý giữ nguyên lr để chỉ khác một yếu tố là trọng số pretrained).", None),
         ("", None), ("Điểm chung", "h"),
         ("Cùng dữ liệu (587/126/126 ảnh), cùng tiền xử lý và augmentation, cùng 14 concept, cùng seed 42/43/44, cùng cách chọn mô hình "
          "(val macro-F1 rồi val CE) và cách chọn ngưỡng concept (trên val, áp dụng cho test).", None),
         ("Khối được mở ở GĐ2 (quy tắc chung: mọi khối ở độ phân giải thấp nhất, stride 32):", None)]
    for bb in ORDER:
        L.append((f"   • {NAME[bb]}: {', '.join(tl['unfreeze_stride32'][bb])}", None))
    L += [("", None), ("Cách đọc", "h"),
          ("Mỗi ô là trung bình qua 3 seed; std = độ lệch chuẩn giữa 3 seed; n = số lượt đã có. "
           "Cột 'X − A': xanh = chiến lược X cao hơn cách đang dùng, đỏ = thấp hơn. Nếu |chênh lệch| nhỏ hơn std thì nằm trong mức dao động do seed.", None),
          ("", None), ("Số liệu tóm tắt (tính tự động, test macro-F1)", "h")]
    for mode, mlab in MODES:
        for code in present:
            m, s, n = ms(df.loc[(df["mode"] == mode) & (df.strategy == code), "test_f1"])
            L.append((f"• {mlab} – {SHORT[code]}: {m:.4f} ± {s:.4f} (trung bình {n} lượt)", None))
        for code in present:
            if code == "A":
                continue
            d = paired(df, mode, "A", code)
            if len(d):
                p = wilcoxon_p(d)
                L.append((f"   {mlab}: {code} − A = {d.mean():+.4f} trên {len(d)} cặp (backbone, seed); {code} cao hơn ở {(d > 0).sum()} cặp, "
                          f"thấp hơn ở {(d < 0).sum()} cặp" + (f"; Wilcoxon p = {p:.3f}" if not np.isnan(p) else ""), None))
    L += [("", None), ("Các sheet", "h"),
          ("Tong_hop: bảng + biểu đồ theo backbone | Chi_phi: thời gian, số epoch, số tham số được học | Tung_luot: từng lượt | "
           "AUC_tung_concept: CBM | F1_theo_lop: từng bệnh | Bieu_do: hình cho luận văn", None)]
    for i, (t, kind) in enumerate(L, 1):
        c = ws.cell(i, 1, t)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if kind:
            c.font = Font(bold=True, size=13 if kind == "h1" else 11)
    ws.column_dimensions["A"].width = 150

    # ===== Tong_hop
    ws = wb.create_sheet("Tong_hop")
    blocks = [("baseline", "test_f1", "Baseline – Test macro-F1"), ("cbm", "test_f1", "CBM – Test macro-F1"),
              ("cbm", "test_cauc", "CBM – AUC concept trung bình (14 concept, test)"),
              ("baseline", "test_acc", "Baseline – Test accuracy"), ("cbm", "test_acc", "CBM – Test accuracy"),
              ("baseline", "val_f1", "Baseline – Val macro-F1"), ("cbm", "val_f1", "CBM – Val macro-F1"),
              ("cbm", "test_cf1", "CBM – F1 concept trung bình (test)")]
    r = 1
    charts = []
    for mode, col, title in blocks:
        r_next, hr, first, last, mean_cols = metric_block(ws, r, df, mode, col, title, present)
        if len(charts) < 3:
            charts.append((title, hr, first, last, mean_cols, col))
        r = r_next
    widths(ws, [20] + [12, 7, 4] * len(present) + [10] * (len(present) - 1))
    cols = [s[3] for s in STRATS if s[0] in present]
    anchor_col = get_column_letter(3 * len(present) + len(present) + 2)
    for i, (title, hr, first, last, mean_cols, col) in enumerate(charts):
        lo = 0.8 if col == "test_cauc" else 0.4
        bar_chart(ws, title, "AUC" if col == "test_cauc" else "Macro-F1", 1, [mean_cols[c] for c in present], hr, first, last,
                  f"{anchor_col}{1 + i * 19}", cols, ymin=lo, ymax=1.0)

    # ===== Chi_phi
    ws = wb.create_sheet("Chi_phi")
    ws.cell(1, 1, "Chi phí huấn luyện (trung bình 3 seed). B tính cả GĐ1 + GĐ2. Tham số được học: B = ở GĐ2; D = chỉ phần đầu.").font = Font(bold=True)
    head = ["Backbone", "Loại"]
    for code in present:
        head += [f"{SHORT[code]}\nphút train", "epoch chạy", "tham số được học"]
    header(ws, 3, head)
    ws.row_dimensions[3].height = 45
    r = 3
    for bb in ORDER:
        for mode, mlab in MODES:
            g = df[(df.backbone == bb) & (df["mode"] == mode)]
            if not len(g):
                continue
            r += 1
            put(ws, r, 1, NAME[bb], bold=True); put(ws, r, 2, mlab)
            c = 3
            for code in present:
                h = g[g.strategy == code]
                put(ws, r, c, ms(h.train_min)[0], "0.0"); put(ws, r, c + 1, ms(h.epochs_run)[0], "0.0")
                put(ws, r, c + 2, ms(h.trainable_params)[0], "#,##0")
                c += 3
    widths(ws, [16, 9] + [12, 9, 14] * len(present))

    # ===== Tung_luot
    ws = wb.create_sheet("Tung_luot")
    cols_raw = [("Backbone", "backbone"), ("Loại", "mode"), ("Seed", "seed"), ("Chiến lược", "strategy"),
                ("Test F1", "test_f1"), ("Test acc", "test_acc"), ("Val F1", "val_f1"), ("AUC concept (test)", "test_cauc"),
                ("F1 concept (test)", "test_cf1"), ("Epoch chạy", "epochs_run"), ("Best epoch", "best_epoch"),
                ("Phút train", "train_min"), ("Tham số được học", "trainable_params"), ("Tổng tham số", "parameters")]
    header(ws, 1, [c for c, _ in cols_raw])
    ws.row_dimensions[1].height = 32
    order = {s[0]: i for i, s in enumerate(STRATS)}
    dd = df.assign(_b=df.backbone.map(ORDER.index), _s=df.strategy.map(order)).sort_values(["_b", "mode", "seed", "_s"])
    for i, row in enumerate(dd.itertuples(index=False), 2):
        row = row._asdict()
        for j, (_, k) in enumerate(cols_raw, 1):
            v = row[k]
            if k == "backbone":
                v = NAME[v]
            elif k == "strategy":
                v = SHORT[v]
            fmt = NUM if k.startswith(("test", "val")) else ("0.0" if k == "train_min" else "#,##0")
            put(ws, i, j, v, fmt)
    widths(ws, [16, 9, 6, 22, 9, 9, 9, 10, 10, 8, 8, 9, 14, 14])
    ws.freeze_panes = "E2"

    # ===== AUC_tung_concept
    ws = wb.create_sheet("AUC_tung_concept")
    ws.cell(1, 1, "CBM – AUC từng concept trên test (trung bình 5 backbone × 3 seed).").font = Font(bold=True)
    header(ws, 3, ["Concept"] + [SHORT[c] for c in present] + [f"{c} − A" for c in present if c != "A"])
    ws.row_dimensions[3].height = 32
    piv = pc.groupby(["concept", "strategy"])["auc"].mean() if len(pc) else pd.Series(dtype=float)
    concepts = sorted(pc.concept.unique()) if len(pc) else []
    for i, c in enumerate(concepts, 4):
        put(ws, i, 1, c, bold=True)
        vals = {code: piv.get((c, code), np.nan) for code in present}
        for j, code in enumerate(present, 2):
            put(ws, i, j, vals[code], "0.000")
        j = 2 + len(present)
        for code in present:
            if code != "A":
                put(ws, i, j, vals[code] - vals.get("A", np.nan), delta=True)
                j += 1
    widths(ws, [9] + [18] * len(present) + [9] * (len(present) - 1))
    if concepts:
        bar_chart(ws, "AUC từng concept (CBM, test)", "AUC", 1, list(range(2, 2 + len(present))), 3, 4, 3 + len(concepts),
                  f"{get_column_letter(3 + 2 * len(present))}3", cols, ymin=0.5, ymax=1.0)

    # ===== F1_theo_lop
    ws = wb.create_sheet("F1_theo_lop")
    ws.cell(1, 1, "F1 từng bệnh trên test (trung bình 5 backbone × 3 seed).").font = Font(bold=True)
    r = 3
    for mode, mlab in MODES:
        ws.cell(r, 1, mlab).font = Font(bold=True, size=12)
        header(ws, r + 1, ["Lớp"] + [SHORT[c] for c in present] + [f"{c} − A" for c in present if c != "A"])
        g = pcls[pcls["mode"] == mode].groupby(["class", "strategy"])["f1"].mean() if len(pcls) else pd.Series(dtype=float)
        for i, cls in enumerate(CLASS_VN, r + 2):
            put(ws, i, 1, CLASS_VN[cls], bold=True)
            vals = {code: g.get((cls, code), np.nan) for code in present}
            for j, code in enumerate(present, 2):
                put(ws, i, j, vals[code], NUM)
            j = 2 + len(present)
            for code in present:
                if code != "A":
                    put(ws, i, j, vals[code] - vals.get("A", np.nan), delta=True)
                    j += 1
        r += len(CLASS_VN) + 4
    widths(ws, [12] + [18] * len(present) + [9] * (len(present) - 1))

    # ===== Bieu_do
    ws = wb.create_sheet("Bieu_do")
    ws.cell(1, 1, "Hình cho luận văn (lưu riêng: so_sanh/so_sanh_transfer_learning.png). Thanh sai số = độ lệch chuẩn qua 3 seed.").font = Font(bold=True)
    img = XLImage(str(png))
    img.width, img.height = img.width * 0.33, img.height * 0.33
    ws.add_image(img, "A3")

    xlsx = out_dir / "SoSanh_TransferLearning.xlsx"
    try:
        wb.save(xlsx)
    except PermissionError:
        xlsx = out_dir / "SoSanh_TransferLearning_moi.xlsx"
        wb.save(xlsx)
        print("File Excel cu dang mo, da luu ban moi:", xlsx)
    print("So luot moi chien luoc:", df.groupby("strategy").size().to_dict())
    print("Da luu:", xlsx)
    print("Da luu:", png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
