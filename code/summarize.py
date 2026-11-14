"""Tong hop cac luot da xong (chi doc runs/*/metrics.json, khong dung toi mo hinh).
Chay bat cu luc nao: python summarize.py
Tao trong runs/: tat_ca_luot.csv, tong_hop_mean_std.csv, so_sanh_cap.csv, concept_auc_theo_backbone.csv,
                  BANG_SO_SANH.md, so_sanh.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ORDER = ["resnet18", "densenet121", "efficientnet_b0", "convnext_tiny", "mobilenet_v2"]


def fmt(m, s):
    return "-" if pd.isna(m) else (f"{m:.4f}" if pd.isna(s) else f"{m:.4f} ± {s:.4f}")


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    common = json.loads((HERE / "configs" / "common.json").read_text(encoding="utf-8"))
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(common["runs_root"])
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/metrics.json"))]
    if not recs:
        print("Chua co luot nao xong trong", root)
        return
    df = pd.DataFrame(recs)
    df["bb_order"] = df["backbone"].map(ORDER.index)
    df = df.sort_values(["bb_order", "mode", "seed"]).drop(columns="bb_order")
    df.to_csv(root / "tat_ca_luot.csv", index=False)

    metrics = ["val_acc", "val_f1", "test_acc", "test_f1", "test_concept_auc", "test_concept_f1", "test_rule_violation",
               "epochs_run", "best_epoch", "training_seconds", "test_ms_per_image", "parameters"]
    metrics = [m for m in metrics if m in df]
    g = df.groupby(["backbone", "mode"], sort=False)
    agg = g[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.insert(0, "so_seed", g.size())
    agg = agg.reset_index()
    agg.to_csv(root / "tong_hop_mean_std.csv", index=False)

    rows = []
    for _, r in agg.iterrows():
        rows.append({"Backbone": r["backbone"], "Loai": r["mode"], "Seed": int(r["so_seed"]),
                     "Val acc": fmt(r["val_acc_mean"], r["val_acc_std"]), "Val F1": fmt(r["val_f1_mean"], r["val_f1_std"]),
                     "Test acc": fmt(r["test_acc_mean"], r["test_acc_std"]), "Test F1": fmt(r["test_f1_mean"], r["test_f1_std"]),
                     "Concept AUC (test)": fmt(r.get("test_concept_auc_mean"), r.get("test_concept_auc_std")),
                     "Concept F1 (test)": fmt(r.get("test_concept_f1_mean"), r.get("test_concept_f1_std")),
                     "Tham so (M)": f"{r['parameters_mean'] / 1e6:.1f}",
                     "Phut train": f"{r['training_seconds_mean'] / 60:.1f}"})
    main_tab = pd.DataFrame(rows)

    pair = []
    for bb in ORDER:
        b = agg[(agg.backbone == bb) & (agg["mode"] == "baseline")]
        c = agg[(agg.backbone == bb) & (agg["mode"] == "cbm")]
        if len(b) and len(c):
            pair.append({"Backbone": bb, "Baseline test F1": f"{b.test_f1_mean.iloc[0]:.4f}", "CBM test F1": f"{c.test_f1_mean.iloc[0]:.4f}",
                         "Chenh lech F1 (CBM - baseline)": f"{c.test_f1_mean.iloc[0] - b.test_f1_mean.iloc[0]:+.4f}",
                         "Baseline test acc": f"{b.test_acc_mean.iloc[0]:.4f}", "CBM test acc": f"{c.test_acc_mean.iloc[0]:.4f}",
                         "Chenh lech acc": f"{c.test_acc_mean.iloc[0] - b.test_acc_mean.iloc[0]:+.4f}"})
    pair_tab = pd.DataFrame(pair)
    pair_tab.to_csv(root / "so_sanh_cap.csv", index=False)

    cm = []
    for p in sorted(root.glob("*_cbm_seed*/test_concept_metrics.csv")):
        if (p.parent / "metrics.json").exists():
            t = pd.read_csv(p)
            t["backbone"] = p.parent.name.split("_cbm_")[0]
            cm.append(t)
    auc_tab = None
    if cm:
        cm = pd.concat(cm)
        auc_tab = cm.pivot_table(index="concept", columns="backbone", values="auc", aggfunc="mean")
        auc_tab = auc_tab[[b for b in ORDER if b in auc_tab.columns]]
        auc_tab.to_csv(root / "concept_auc_theo_backbone.csv")

    md = ["# Bang so sanh baseline va CBM (trung binh ± do lech chuan qua cac seed)", "",
          f"So luot da xong: {len(df)} / 30", "", md_table(main_tab), ""]
    if len(pair_tab):
        md += ["## So sanh tung cap (trung binh test)", "", md_table(pair_tab), ""]
    if auc_tab is not None:
        t = auc_tab.round(3).reset_index()
        md += ["## AUC tung concept tren test (CBM, trung binh cac seed)", "", md_table(t), ""]
    (root / "BANG_SO_SANH.md").write_text("\n".join(md), encoding="utf-8")

    plot = agg.pivot(index="backbone", columns="mode", values="test_f1_mean").reindex([b for b in ORDER if b in set(agg.backbone)])
    err = agg.pivot(index="backbone", columns="mode", values="test_f1_std").reindex(plot.index).fillna(0)
    ax = plot.plot.bar(yerr=err, rot=15, figsize=(9, 4.5), ylim=(0.5, 1.0), capsize=3)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Baseline vs CBM (trung binh ± std qua seed)")
    plt.tight_layout()
    plt.savefig(root / "so_sanh.png", dpi=150)
    print("\n".join(md))


if __name__ == "__main__":
    main()
