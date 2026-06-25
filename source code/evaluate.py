"""
evaluate.py
===========
Evaluate the full Fuzzy Weighted Ensemble pipeline on the test split.
Compares baseline (equal weights) vs fuzzy-weighted performance.

Run AFTER train_experts.py:
  python evaluate.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score
)

from fuzzy_engine import FuzzyContext, compute_fuzzy_weights, compute_fuzzy_confidence, classify_result

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

DATA_PATH   = "URL-Phish.csv"
MODEL_DIR   = "models"
DROP_COLS   = ["url", "dom", "tld", "label"]
TARGET_COL  = "label"
RANDOM_SEED = 42
TEST_SIZE   = 0.2


def load_test_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=DROP_COLS).fillna(0)
    y = df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return X_test[feature_cols], y_test


def load_models():
    return {
        "RF":  joblib.load(os.path.join(MODEL_DIR, "rf_expert.pkl")),
        "XGB": joblib.load(os.path.join(MODEL_DIR, "xgb_expert.pkl")),
        "SVM": joblib.load(os.path.join(MODEL_DIR, "svm_expert.pkl")),
    }


def get_tier1_probas(models, X_test):
    return {
        name: model.predict_proba(X_test)[:, 1]
        for name, model in models.items()
    }


def baseline_ensemble(probas) -> np.ndarray:
    """Simple equal-weight average"""
    stacked = np.stack(list(probas.values()), axis=1)
    return stacked.mean(axis=1)


def fuzzy_ensemble(probas, X_test: pd.DataFrame) -> np.ndarray:
    """Fuzzy-weighted ensemble using context from feature columns"""
    scores = []
    for i in range(len(X_test)):
        row = X_test.iloc[i]

        ctx = FuzzyContext(
            domain_age_days   = int(row.get("domain_age_days",  -1)),
            ssl_validity_days = int(row.get("ssl_validity_days", -1)),
            has_ip            = int(row.get("has_ip_address",    0)),
            uses_https        = int(row.get("uses_https",        0)),
            has_dns           = int(row.get("has_dns_record",    1)),
            rf_pred           = int(probas["RF"][i]  > 0.5),
            xgb_pred          = int(probas["XGB"][i] > 0.5),
            svm_pred          = int(probas["SVM"][i] > 0.5),
            rf_prob           = float(probas["RF"][i]),
            xgb_prob          = float(probas["XGB"][i]),
            svm_prob          = float(probas["SVM"][i]),
        )
        weights = compute_fuzzy_weights(ctx)
        P = compute_fuzzy_confidence(ctx, weights)
        scores.append(P)
    return np.array(scores)


def evaluate(y_true, y_proba, label: str):
    y_pred = (y_proba > 0.5).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_proba)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'─'*45}")
    print(f"  {label}")
    print(f"{'─'*45}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(classification_report(y_true, y_pred,
                                target_names=["Legitimate", "Phishing"]))
    return {"acc": acc, "f1": f1, "auc": auc}


def main():
    print("\n" + "═" * 45)
    print("  Fuzzy Ensemble — Evaluation Report")
    print("═" * 45)

    print("\n[1/4] Loading test data and models...")
    X_test, y_test = load_test_data()
    models = load_models()

    print("[2/4] Getting Tier-1 probabilities...")
    probas = get_tier1_probas(models, X_test)

    print("[3/4] Computing ensemble scores...")
    baseline_scores = baseline_ensemble(probas)
    fuzzy_scores    = fuzzy_ensemble(probas, X_test)

    print("[4/4] Evaluating...\n")

    # Individual models
    for name, prob in probas.items():
        evaluate(y_test, prob, f"Expert: {name}")

    # Baselines
    baseline_res = evaluate(y_test, baseline_scores, "Baseline (Equal Weights)")
    fuzzy_res    = evaluate(y_test, fuzzy_scores,    "Fuzzy Weighted Ensemble ★")

    # Summary
    print("\n" + "═" * 45)
    print("  IMPROVEMENT SUMMARY")
    print("═" * 45)
    for metric in ["acc", "f1", "auc"]:
        delta = fuzzy_res[metric] - baseline_res[metric]
        sign  = "+" if delta >= 0 else ""
        print(f"  {metric.upper():<8}: {baseline_res[metric]:.4f} → {fuzzy_res[metric]:.4f}  ({sign}{delta:.4f})")
    print()


if __name__ == "__main__":
    main()
