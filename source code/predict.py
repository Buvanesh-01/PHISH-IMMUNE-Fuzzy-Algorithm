"""
predict.py
==========
Full Inference Pipeline — Fuzzy Weighted Ensemble

Given a raw URL, this script:
  1. Extracts 22 numerical features (feature_extractor.py)
  2. Runs Tier-1 parallel classification (RF, XGB, SVM)
  3. Applies Tier-2 fuzzy weighting (fuzzy_engine.py)
  4. Outputs final probability P and verdict

Usage:
  python predict.py --url "http://suspicious-site.com/login"
  python predict.py --file urls.txt
  python predict.py --demo
"""

import os
import sys
import argparse
import time
import joblib
import numpy as np
import pandas as pd

from feature_extractor import extract_features
from fuzzy_engine import (
    FuzzyContext, compute_fuzzy_weights,
    compute_fuzzy_confidence, classify_result
)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

MODEL_DIR = "models"
MODELS = {
    "RF":  os.path.join(MODEL_DIR, "rf_expert.pkl"),
    "XGB": os.path.join(MODEL_DIR, "xgb_expert.pkl"),
    "SVM": os.path.join(MODEL_DIR, "svm_expert.pkl"),
}
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────

def load_models():
    models = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run `python train_experts.py` first."
            )
        models[name] = joblib.load(path)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return models, feature_cols


# ─────────────────────────────────────────────
#  TIER-1 PREDICTION
# ─────────────────────────────────────────────

def tier1_predict(models, feature_cols, features_dict: dict) -> dict:
    """
    Run all three expert classifiers.
    Returns predictions and probabilities.
    """
    # Build feature vector in correct column order
    print("Extracted features:", features_dict.keys())
    print("Expected features:", feature_cols)
    X = pd.DataFrame([features_dict])[feature_cols]

    results = {}
    for name, model in models.items():
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0]  # [prob_legit, prob_phish]
        results[name] = {
            "pred":        int(pred),
            "prob_phish":  round(float(proba[1]), 4),
            "prob_legit":  round(float(proba[0]), 4),
        }
    return results


# ─────────────────────────────────────────────
#  TIER-2 FUZZY WEIGHTING
# ─────────────────────────────────────────────

def tier2_fuzzy(features_dict: dict, tier1_results: dict) -> dict:
    """
    Build FuzzyContext and compute dynamic weights + final score.
    """
    ctx = FuzzyContext(
        domain_age_days    = features_dict.get("domain_age_days", -1),
        ssl_validity_days  = features_dict.get("ssl_validity_days", -1),
        has_ip             = features_dict.get("has_ip_address", 0),
        uses_https         = features_dict.get("uses_https", 0),
        has_dns            = features_dict.get("has_dns_record", 1),
        rf_pred            = tier1_results["RF"]["pred"],
        xgb_pred           = tier1_results["XGB"]["pred"],
        svm_pred           = tier1_results["SVM"]["pred"],
        rf_prob            = tier1_results["RF"]["prob_phish"],
        xgb_prob           = tier1_results["XGB"]["prob_phish"],
        svm_prob           = tier1_results["SVM"]["prob_phish"],
    )

    weights = compute_fuzzy_weights(ctx)
    P       = compute_fuzzy_confidence(ctx, weights)
    verdict = classify_result(P)

    return {
        "weights": weights,
        "P":       P,
        "verdict": verdict,
        "context": ctx,
    }


# ─────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────

def predict_url(url: str, models: dict, feature_cols: list, verbose: bool = True) -> dict:
    t0 = time.time()

    # Step 1: Feature extraction
    features = extract_features(url)

    # Step 2: Tier-1 parallel classification
    t1_results = tier1_predict(models, feature_cols, features)

    # Step 3: Tier-2 fuzzy weighting
    t2_results = tier2_fuzzy(features, t1_results)

    elapsed = round(time.time() - t0, 3)

    result = {
        "url":        url,
        "features":   features,
        "tier1":      t1_results,
        "weights":    t2_results["weights"],
        "P":          t2_results["P"],
        "verdict":    t2_results["verdict"],
        "elapsed_s":  elapsed,
    }

    if verbose:
        print_result(result)

    return result


# ─────────────────────────────────────────────
#  PRETTY PRINT
# ─────────────────────────────────────────────

VERDICT_ICONS = {
    "PHISHING":   "🚨",
    "SUSPICIOUS": "⚠️ ",
    "LEGITIMATE": "✅",
}

def print_result(r: dict):
    verdict = r["verdict"]
    icon    = VERDICT_ICONS.get(verdict, "?")

    print("\n" + "═" * 60)
    print(f"  URL      : {r['url']}")
    print("─" * 60)
    print("  TIER-1  (Expert Predictions)")
    for name, v in r["tier1"].items():
        label = "Phish  " if v["pred"] == 1 else "Legit  "
        bar   = "█" * int(v["prob_phish"] * 20)
        print(f"    {name:<5}: {label}  P(phish)={v['prob_phish']:.3f}  {bar}")
    print("─" * 60)
    print("  TIER-2  (Fuzzy Weights)")
    w = r["weights"]
    print(f"    RF={w['RF']:.3f}  XGB={w['XGB']:.3f}  SVM={w['SVM']:.3f}")
    print("─" * 60)
    print(f"  SCORE P  : {r['P']:.4f}")
    print(f"  VERDICT  : {icon} {verdict}")
    print(f"  TIME     : {r['elapsed_s']}s")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────
#  BATCH PREDICTION
# ─────────────────────────────────────────────

def predict_batch(urls: list, models: dict, feature_cols: list) -> pd.DataFrame:
    rows = []
    for url in urls:
        try:
            r = predict_url(url, models, feature_cols, verbose=False)
            rows.append({
                "url":     r["url"],
                "P":       r["P"],
                "verdict": r["verdict"],
                "W_RF":    r["weights"]["RF"],
                "W_XGB":   r["weights"]["XGB"],
                "W_SVM":   r["weights"]["SVM"],
                "RF_prob":  r["tier1"]["RF"]["prob_phish"],
                "XGB_prob": r["tier1"]["XGB"]["prob_phish"],
                "SVM_prob": r["tier1"]["SVM"]["prob_phish"],
            })
        except Exception as e:
            rows.append({"url": url, "verdict": "ERROR", "P": -1})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  DEMO MODE (no real network calls)
# ─────────────────────────────────────────────

DEMO_URLS = [
    "https://www.google.com/search?q=weather",
    "http://secure-login.paypa1.com/verify?token=abc123",
    "https://amazon.com/orders",
    "http://192.168.1.1/admin/login.php",
    "http://free-iphone-winner.tk/claim?id=99382",
]


# ─────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fuzzy Weighted Ensemble — URL Phishing Detector"
    )
    parser.add_argument("--url",   type=str, help="Single URL to classify")
    parser.add_argument("--file",  type=str, help="Text file with one URL per line")
    parser.add_argument("--demo",  action="store_true", help="Run on built-in demo URLs")
    parser.add_argument("--out",   type=str, default=None, help="Save batch results to CSV")
    args = parser.parse_args()

    print("\n  Loading models...")
    models, feature_cols = load_models()
    print("  ✅ Models loaded.\n")

    if args.url:
        predict_url(args.url, models, feature_cols)

    elif args.file:
        with open(args.file) as f:
            urls = [line.strip() for line in f if line.strip()]
        df = predict_batch(urls, models, feature_cols)
        print(df.to_string(index=False))
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"\n  Results saved → {args.out}")

    elif args.demo:
        print("  Running on demo URLs (features computed locally)...\n")
        for url in DEMO_URLS:
            predict_url(url, models, feature_cols)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
