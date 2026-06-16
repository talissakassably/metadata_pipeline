# -*- coding: utf-8 -*-
r"""
Machine-learning add-on for the LEC discovery layer.

Purpose
-------
Add a small, defensible ML analysis to the neuroscience discovery layer.

Biological question
-------------------
Can LEC population temporal-organization features predict behavioral context?

Interpretation
--------------
This is NOT meant to be a huge deep-learning model.
It is a compact validation analysis:

If simple ML can classify context from LEC temporal-drift features above chance,
then behavioral context leaves a measurable signature in the temporal organization
of LEC population activity.

Inputs
------
Run this AFTER:

case_studies\nwb_lec_context_drift_discovery.py

Required files:
outputs\nwb_lec_context_drift_discovery\
├── lec_context_drift_session_summary.csv
├── lec_context_distance_lag_curves.csv
└── lec_context_drift_repeated_subsampling.csv

Run
---
python case_studies\nwb_lec_context_drift_ml.py ^
  --discovery_dir outputs\nwb_lec_context_drift_discovery ^
  --output_dir outputs\nwb_lec_context_drift_ml

Safer binary run:
python case_studies\nwb_lec_context_drift_ml.py ^
  --discovery_dir outputs\nwb_lec_context_drift_discovery ^
  --output_dir outputs\nwb_lec_context_drift_ml_binary ^
  --mode binary_event_vs_open
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_inputs(discovery_dir):
    discovery_dir = Path(discovery_dir)
    session = pd.read_csv(discovery_dir / "lec_context_drift_session_summary.csv")
    curves_path = discovery_dir / "lec_context_distance_lag_curves.csv"
    repeats_path = discovery_dir / "lec_context_drift_repeated_subsampling.csv"

    curves = pd.read_csv(curves_path) if curves_path.exists() else pd.DataFrame()
    repeats = pd.read_csv(repeats_path) if repeats_path.exists() else pd.DataFrame()
    return session, curves, repeats


def curve_features(curves):
    if curves.empty:
        return pd.DataFrame(columns=["session_id"])

    rows = []
    for sid, sub in curves.groupby("session_id"):
        sub = sub.sort_values("lag_value_s")
        y = pd.to_numeric(sub["mean_distance"], errors="coerce").values
        x = pd.to_numeric(sub["lag_value_s"], errors="coerce").values
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if len(y) < 3:
            continue

        early = y[:max(2, len(y)//4)]
        late = y[-max(2, len(y)//4):]
        auc = np.trapz(y, x) / (x.max() - x.min()) if x.max() > x.min() else np.nan
        slope = np.polyfit(x, y, 1)[0] if len(x) >= 3 else np.nan

        rows.append({
            "session_id": sid,
            "curve_mean_distance": float(np.mean(y)),
            "curve_initial_distance": float(y[0]),
            "curve_final_distance": float(y[-1]),
            "curve_early_mean": float(np.mean(early)),
            "curve_late_mean": float(np.mean(late)),
            "curve_late_minus_early": float(np.mean(late) - np.mean(early)),
            "curve_auc": float(auc),
            "curve_slope_raw": float(slope),
        })

    return pd.DataFrame(rows)


def repeat_features(repeats):
    if repeats.empty:
        return pd.DataFrame(columns=["session_id"])

    rows = []
    for sid, sub in repeats.groupby("session_id"):
        vals = pd.to_numeric(sub["toi"], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        rows.append({
            "session_id": sid,
            "repeat_toi_mean": float(np.mean(vals)),
            "repeat_toi_median": float(np.median(vals)),
            "repeat_toi_std": float(np.std(vals)),
            "repeat_toi_q25": float(np.percentile(vals, 25)),
            "repeat_toi_q75": float(np.percentile(vals, 75)),
            "repeat_toi_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        })
    return pd.DataFrame(rows)


def build_feature_table(session, curves, repeats, mode):
    ok = session[session["status"].eq("ok")].copy()

    # Keep only interpretable contexts.
    ok = ok[ok["context"].isin(["open_field", "object_context", "sequence_task", "sleep"])].copy()

    cf = curve_features(curves)
    rf = repeat_features(repeats)

    df = ok.merge(cf, on="session_id", how="left").merge(rf, on="session_id", how="left")

    # Base features from discovery layer.
    feature_cols = [
        "toi_median",
        "toi_mean",
        "toi_std",
        "duration_s_median",
        "n_bins_median",
        "n_units_available",
        "curve_mean_distance",
        "curve_initial_distance",
        "curve_final_distance",
        "curve_early_mean",
        "curve_late_mean",
        "curve_late_minus_early",
        "curve_auc",
        "curve_slope_raw",
        "repeat_toi_mean",
        "repeat_toi_median",
        "repeat_toi_std",
        "repeat_toi_iqr",
    ]

    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if mode == "binary_event_vs_open":
        # Make a cleaner binary ML question.
        # open_field vs event/context structured.
        df = df[df["context"].isin(["open_field", "object_context", "sequence_task"])].copy()
        df["ml_label"] = np.where(df["context"].eq("open_field"), "open_field", "event_or_object_context")
    elif mode == "multiclass_no_sequence":
        # Exclude sequence because n=2 is too small for stable ML.
        df = df[df["context"].isin(["open_field", "object_context", "sleep"])].copy()
        df["ml_label"] = df["context"]
    else:
        df["ml_label"] = df["context"]

    # Drop features that are all missing.
    feature_cols = [c for c in feature_cols if df[c].notna().sum() >= 3]

    # Median imputation.
    for c in feature_cols:
        med = df[c].median()
        if np.isfinite(med):
            df[c] = df[c].fillna(med)
        else:
            df[c] = df[c].fillna(0)

    return df, feature_cols


def run_ml(df, feature_cols, output_dir, n_permutations=200, random_state=42):
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, LeaveOneOut
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
        from sklearn.inspection import permutation_importance
    except Exception as e:
        raise RuntimeError("scikit-learn is required. Install with: python -m pip install scikit-learn") from e

    X = df[feature_cols].values.astype(float)
    y_text = df["ml_label"].astype(str).values

    # Remove labels with <2 sessions; otherwise CV breaks.
    counts = pd.Series(y_text).value_counts()
    keep_labels = counts[counts >= 2].index
    keep = np.isin(y_text, keep_labels)
    X = X[keep]
    y_text = y_text[keep]
    df_used = df.loc[keep].copy()

    le = LabelEncoder()
    y = le.fit_transform(y_text)
    labels = list(le.classes_)

    if len(labels) < 2:
        raise RuntimeError("Not enough classes with at least 2 samples.")

    min_class = pd.Series(y).value_counts().min()
    if min_class >= 3:
        cv = StratifiedKFold(n_splits=min(5, int(min_class)), shuffle=True, random_state=random_state)
        cv_name = f"StratifiedKFold n_splits={min(5, int(min_class))}"
    else:
        cv = LeaveOneOut()
        cv_name = "LeaveOneOut"

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            max_depth=3,
        ),
    }

    summary_rows = []
    prediction_tables = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
        y_pred = cross_val_predict(model, X, y, cv=cv)

        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)

        # Permutation null: shuffle labels and recompute CV balanced accuracy.
        rng = np.random.default_rng(random_state)
        null_scores = []
        for _ in range(n_permutations):
            y_perm = rng.permutation(y)
            try:
                s = cross_val_score(model, X, y_perm, cv=cv, scoring="balanced_accuracy")
                null_scores.append(np.mean(s))
            except Exception:
                pass

        null_scores = np.asarray(null_scores, dtype=float)
        p_perm = float((np.sum(null_scores >= bal_acc) + 1) / (len(null_scores) + 1)) if len(null_scores) else np.nan

        summary_rows.append({
            "model": name,
            "cv": cv_name,
            "n_samples": len(y),
            "n_classes": len(labels),
            "classes": ";".join(labels),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "cv_balanced_accuracy_mean": float(np.mean(scores)),
            "cv_balanced_accuracy_std": float(np.std(scores)),
            "permutation_p_value": p_perm,
            "permutation_null_mean": float(np.mean(null_scores)) if len(null_scores) else np.nan,
            "permutation_null_std": float(np.std(null_scores)) if len(null_scores) else np.nan,
        })

        pred_df = df_used[["session_id", "subject_id", "context", "ml_label"]].copy()
        pred_df["predicted_label"] = le.inverse_transform(y_pred)
        pred_df["correct"] = pred_df["ml_label"].eq(pred_df["predicted_label"])
        prediction_tables[name] = pred_df
        pred_df.to_csv(output_dir / f"predictions_{name}.csv", index=False)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred, labels=np.arange(len(labels)))
        plt.figure(figsize=(5.5, 5))
        plt.imshow(cm, aspect="auto")
        plt.colorbar(label="sessions")
        plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{name}: context classification")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(fig_dir / f"confusion_matrix_{name}.png", dpi=300)
        plt.close()

        # Null distribution
        if len(null_scores):
            plt.figure(figsize=(6, 4))
            plt.hist(null_scores, bins=20, alpha=0.8)
            plt.axvline(bal_acc, linestyle="--", label=f"observed={bal_acc:.2f}")
            plt.xlabel("Balanced accuracy under shuffled labels")
            plt.ylabel("Permutations")
            plt.title(f"{name}: permutation test")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / f"permutation_test_{name}.png", dpi=300)
            plt.close()

        # Fit full model for feature importance.
        model.fit(X, y)
        if name == "random_forest":
            importances = model.feature_importances_
        else:
            clf = model.named_steps["clf"]
            coef = clf.coef_
            importances = np.mean(np.abs(coef), axis=0)

        imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
        imp = imp.sort_values("importance", ascending=False)
        imp.to_csv(output_dir / f"feature_importance_{name}.csv", index=False)

        top = imp.head(12).iloc[::-1]
        plt.figure(figsize=(7, 4.8))
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Importance")
        plt.title(f"{name}: top temporal-drift features")
        plt.tight_layout()
        plt.savefig(fig_dir / f"feature_importance_{name}.png", dpi=300)
        plt.close()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "ml_context_classification_summary.csv", index=False)

    df_used.to_csv(output_dir / "ml_feature_table_used.csv", index=False)

    return summary, labels


def write_report(path, summary, labels, mode):
    lines = [
        "# ML layer: decoding behavioral context from LEC temporal organization",
        "",
        "## Question",
        "",
        "Can LEC temporal-organization features predict behavioral context?",
        "",
        "## Why this matters",
        "",
        "This adds a machine-learning validation layer to the discovery analysis. If a simple classifier predicts context above permutation baseline, then behavioral context leaves a measurable signature in LEC temporal drift features.",
        "",
        f"## ML mode",
        "",
        f"`{mode}`",
        "",
        "## Classes",
        "",
        ", ".join(labels),
        "",
        "## Results",
        "",
    ]

    for _, row in summary.iterrows():
        lines.append(f"### {row['model']}")
        lines.append(f"- CV: {row['cv']}")
        lines.append(f"- Samples: {int(row['n_samples'])}")
        lines.append(f"- Balanced accuracy: {row['balanced_accuracy']:.3f}")
        lines.append(f"- CV balanced accuracy mean ± SD: {row['cv_balanced_accuracy_mean']:.3f} ± {row['cv_balanced_accuracy_std']:.3f}")
        lines.append(f"- Permutation p-value: {row['permutation_p_value']:.4g}")
        lines.append("")

    lines += [
        "## Recommended interpretation",
        "",
        "Use this as a validation analysis, not as the main biological claim. The main biological result remains the context-dependent TOI difference. The ML result asks whether temporal-drift features are informative enough to recover behavioral context.",
        "",
        "## Best sentence for the report",
        "",
        "A simple machine-learning classifier trained on LEC temporal-organization features was used as a validation step. Above-chance context prediction would indicate that behavioral context is encoded in the temporal structure of population activity, supporting the interpretation that event-structured experience modulates LEC temporal drift.",
    ]

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run ML add-on for LEC temporal drift discovery layer.")
    parser.add_argument("--discovery_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/nwb_lec_context_drift_ml")
    parser.add_argument("--mode", default="binary_event_vs_open",
                        choices=["binary_event_vs_open", "multiclass_no_sequence", "multiclass_all"])
    parser.add_argument("--n_permutations", type=int, default=200)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    session, curves, repeats = load_inputs(args.discovery_dir)
    features, feature_cols = build_feature_table(session, curves, repeats, args.mode)

    features.to_csv(out / "ml_feature_table_all_candidates.csv", index=False)

    summary, labels = run_ml(
        features,
        feature_cols,
        out,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )

    write_report(out / "ml_context_classification_report.md", summary, labels, args.mode)

    print("\nDone.")
    print("Output:", out)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
