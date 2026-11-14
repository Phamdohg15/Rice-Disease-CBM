"""Danh gia tren val va test. Nguong concept chon tren VAL (toi da F1 tung concept), ap dung cho test."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)

GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def choose_thresholds(cprob, clab, names):
    th = {}
    for k, n in enumerate(names):
        m = clab[:, k] != 9
        yk, pk = clab[m, k].astype(int), cprob[m, k]
        if yk.sum() == 0 or yk.sum() == len(yk):
            th[n] = 0.5
            continue
        scores = [f1_score(yk, pk >= t, zero_division=0) for t in GRID]
        best = max(scores)
        cands = [t for t, s in zip(GRID, scores) if s == best]
        th[n] = float(min(cands, key=lambda t: abs(t - 0.5)))
    return th


def concept_table(cprob, clab, names, th, auc_min):
    rows = []
    for k, n in enumerate(names):
        m = clab[:, k] != 9
        yk, pk = clab[m, k].astype(int), cprob[m, k]
        pred = (pk >= th[n]).astype(int)
        two = len(np.unique(yk)) == 2
        rows.append({"concept": n, "n_labeled": int(m.sum()), "n1": int(yk.sum()), "threshold": th[n],
                     "auc": float(roc_auc_score(yk, pk)) if two else np.nan,
                     "f1": float(f1_score(yk, pred, zero_division=0)),
                     "acc": float(accuracy_score(yk, pred)),
                     "bal_acc": float(balanced_accuracy_score(yk, pred)) if two else np.nan})
    df = pd.DataFrame(rows)
    df["dat_nguong_ontology"] = df["auc"] >= auc_min
    return df


def rule_violation_rate(cprob, names, th, absence):
    """Ti le anh du doan C16=1 nhung van bat >=1 concept ton thuong."""
    i = names.index(absence)
    abs_pred = cprob[:, i] >= th[absence]
    les = np.stack([cprob[:, k] >= th[n] for k, n in enumerate(names) if n != absence], 1).any(1)
    return float((abs_pred & les).mean()), int(abs_pred.sum())


def task_report(y, prob, classes):
    pred = prob.argmax(1)
    rep = classification_report(y, pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(len(classes))))
    return pd.DataFrame(rep).T, pd.DataFrame(cm, index=classes, columns=classes)


def predictions_table(df_split, data, cfg):
    rows = []
    for j, i in enumerate(data["idx"]):
        r = {"image_id": df_split.loc[i, "image_id"], "true_class": cfg["classes"][data["y"][j]],
             "pred_class": cfg["classes"][int(data["prob"][j].argmax())]}
        r.update({"p_" + c: float(data["prob"][j][k]) for k, c in enumerate(cfg["classes"])})
        if "cprob" in data:
            r.update({"cp_" + c: float(data["cprob"][j][k]) for k, c in enumerate(cfg["concepts"])})
        rows.append(r)
    return pd.DataFrame(rows)
