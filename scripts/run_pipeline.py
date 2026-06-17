"""
Chạy toàn bộ pipeline end-to-end cho 3 thang DASS, theo chế độ C.TASK_MODE:
  "multiclass" | "binary" | "regression"   (đổi trong src/config.py)

  load -> clean -> scoring -> train nhiều model -> đánh giá
  -> lưu metrics, biểu đồ (confusion / scatter), feature importance, model tốt nhất.

Chạy:  python3 scripts/run_pipeline.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import data_loader, evaluate, features, model, scoring


def run_target(target: str, X: pd.DataFrame, df: pd.DataFrame, mode: str) -> dict:
    print(f"\n{'='*60}\nTARGET: {target}  |  mode: {mode}\n{'='*60}")
    y, labels, kind = scoring.make_target(df, target, mode)

    if kind == "clf":
        print("Phân bố lớp:", dict(pd.Series(y).map(
            lambda i: labels[i]).value_counts().reindex(labels)))
        strat = y
    else:
        print(f"Điểm: mean={y.mean():.1f} std={y.std():.1f} range=[{y.min():.0f},{y.max():.0f}]")
        strat = None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=strat, random_state=C.RANDOM_STATE)

    sel = evaluate.SELECTION_METRIC[mode]
    rows, fitted = [], {}
    for name, est in model.make_models(kind).items():
        pipe = model.fit_pipeline(name, est, X_tr, y_tr, kind)
        pred = pipe.predict(X_te)
        if kind == "clf":
            proba = (pipe.predict_proba(X_te)[:, 1]
                     if mode == "binary" and name != "dummy" else None)
            s = evaluate.scores_clf(y_te, pred, proba)
        else:
            s = evaluate.scores_reg(y_te, pred)
        s["model"] = name
        rows.append(s); fitted[name] = (pipe, pred)
        extra = "  ".join(f"{k}={v:.3f}" for k, v in s.items() if k != "model")
        print(f"  {name:<14} {extra}")

    res = pd.DataFrame(rows).set_index("model")
    higher_better = sel != "mae"
    best = res[sel].idxmax() if higher_better else res[sel].idxmin()
    print(f"  -> model tốt nhất ({sel}): {best}")

    # Lưu artefacts cho model tốt nhất
    pipe, pred = fitted[best]
    if kind == "clf":
        evaluate.save_confusion(y_te, pred, labels, f"{target} — {best}",
                                C.FIG_DIR / f"cm_{target}_{mode}.png")
        (C.METRIC_DIR / f"report_{target}_{mode}.txt").write_text(
            evaluate.text_report(y_te, pred, labels))
    else:
        evaluate.save_regression_scatter(
            y_te, pred, f"{target} — {best}", C.FIG_DIR / f"scatter_{target}.png")
    evaluate.save_feature_importance(pipe, target,
                                     C.FIG_DIR / f"importance_{target}_{mode}.png")
    joblib.dump({"pipeline": pipe, "labels": labels, "mode": mode},
                C.MODEL_DIR / f"model_{target}_{mode}.joblib", compress=3)
    res.to_csv(C.METRIC_DIR / f"metrics_{target}_{mode}.csv")

    return {"target": target, "best_model": best,
            "selection_metric": sel, "best_score": float(res.loc[best, sel])}


def main():
    for d in (C.FIG_DIR, C.MODEL_DIR, C.METRIC_DIR):
        d.mkdir(parents=True, exist_ok=True)
    mode = C.TASK_MODE

    print("1) Tải & làm sạch dữ liệu")
    df = data_loader.load_clean()
    print("2) Tính điểm DASS, severity và Big Five")
    df = scoring.build_targets_and_features(df)
    X = features.select_features(df)
    print(f"   X shape: {X.shape}  |  TASK_MODE = {mode!r}")

    summary = [run_target(t, X, df, mode) for t in C.DASS_SUBSCALES]

    evaluate.dump_json({"mode": mode, "results": summary},
                       C.METRIC_DIR / f"summary_{mode}.json")
    print(f"\n{'='*60}\nTỔNG KẾT  (mode = {mode})\n{'='*60}")
    for s in summary:
        print(f"  {s['target']:<11} best={s['best_model']:<14} "
              f"{s['selection_metric']}={s['best_score']:.3f}")
    print("\nXong. Xem outputs/ để biết chi tiết.")


if __name__ == "__main__":
    main()
