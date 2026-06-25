"""
train_experts.py
================
Tier-1: Train the three expert classifiers on URL-Phish dataset.
Saves:
  - rf_expert.pkl
  - xgb_expert.pkl
  - svm_expert.pkl
  - feature_columns.pkl   ← column order for inference
  - training_report.txt
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

DATA_PATH   = "Dataset.csv"
OUTPUT_DIR  = "models"
DROP_COLS   = ["url", "dom", "tld", "label"]
TARGET_COL  = "label"
TEST_SIZE   = 0.2
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  LOAD & PREPARE DATA
# ─────────────────────────────────────────────

def load_data(path: str):
    print(f"[1/5] Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"      Shape: {df.shape}  |  Label dist:\n{df[TARGET_COL].value_counts().to_string()}\n")

    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL]

    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))

    return X, y


# ─────────────────────────────────────────────
#  DEFINE MODELS
# ─────────────────────────────────────────────

def build_models():
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        verbosity=0,
    )

# SVM wrapped in pipeline with optimized settings for large data
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(
            kernel="linear",          # 'linear' is 100x faster than 'rbf' for large datasets
            C=1.0,                    # Standard regularization
            probability=True,         # REQUIRED for the fuzzy engine to get scores
            max_iter=2000,            # Limits training time so it doesn't run forever
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ))
    ])

    return {"RF": rf, "XGB": xgb, "SVM": svm}


# ─────────────────────────────────────────────
#  TRAIN & EVALUATE
# ─────────────────────────────────────────────

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    report_lines = []

    print("[3/5] Training and evaluating experts...\n")
    for name, model in models.items():
        print(f"  ── {name} ──")
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc     = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        cv_acc  = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()

        print(f"     Accuracy : {acc:.4f}")
        print(f"     ROC-AUC  : {roc_auc:.4f}")
        print(f"     CV Acc   : {cv_acc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])}")

        results[name] = {
            "model":   model,
            "acc":     acc,
            "roc_auc": roc_auc,
            "cv_acc":  cv_acc,
        }
        report_lines.append(
            f"=== {name} ===\n"
            f"Accuracy : {acc:.4f}\n"
            f"ROC-AUC  : {roc_auc:.4f}\n"
            f"CV Acc   : {cv_acc:.4f}\n"
            f"{classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])}\n"
        )

    return results, "\n".join(report_lines)


# ─────────────────────────────────────────────
#  SAVE ARTIFACTS
# ─────────────────────────────────────────────

def save_artifacts(results, feature_cols, report_text):
    print("[4/5] Saving model artifacts...")

    name_map = {"RF": "rf_expert", "XGB": "xgb_expert", "SVM": "svm_expert"}
    for name, data in results.items():
        path = os.path.join(OUTPUT_DIR, f"{name_map[name]}.pkl")
        joblib.dump(data["model"], path)
        print(f"      Saved → {path}")

    joblib.dump(list(feature_cols), os.path.join(OUTPUT_DIR, "feature_columns.pkl"))
    print(f"      Saved → {OUTPUT_DIR}/feature_columns.pkl")

    report_path = os.path.join(OUTPUT_DIR, "training_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"      Saved → {report_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  URL Phish Detection — Tier-1 Expert Training")
    print("=" * 55 + "\n")

    X, y = load_data(DATA_PATH)

    print("[2/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"      Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}\n")

    models  = build_models()
    results, report = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    save_artifacts(results, X.columns, report)

    print("\n[5/5] ✅ All Tier-1 experts trained and saved!")
    print(f"      Models in: ./{OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
