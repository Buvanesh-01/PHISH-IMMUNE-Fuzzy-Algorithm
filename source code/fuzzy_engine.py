"""
fuzzy_engine.py
===============
Tier-2: Fuzzy Weighting Layer

Computes dynamic weights for each classifier based on fuzzy rules
derived from URL-specific context features.

Fuzzy Membership Functions:
  - domain_age  → very_low / low / medium / high
  - ssl_days    → invalid / low / valid / strong
  - confidence  → low / medium / high

Fuzzy Rules (examples):
  R1: IF domain_age IS very_low AND RF says Phish  → W_RF = HIGH
  R2: IF ssl IS valid AND SVM says Legitimate       → W_SVM = HIGH
  R3: IF all three models agree                     → confidence = MAXIMUM
  R4: IF domain_age IS high AND ssl IS strong       → W_XGB = MEDIUM, W_SVM = HIGH
  R5: IF has_ip_address IS true                     → W_RF = VERY_HIGH
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


# ─────────────────────────────────────────────
#  MEMBERSHIP FUNCTIONS (Trapezoidal / Triangular)
# ─────────────────────────────────────────────

def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership: rises a→b, flat b→c, falls c→d"""
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:  # c < x < d
        return (d - x) / (d - c)


def triangle(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership: rises a→b, falls b→c"""
    if x <= a or x >= c:
        return 0.0
    elif x == b:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


# ─────────────────────────────────────────────
#  DOMAIN AGE MEMBERSHIP
# ─────────────────────────────────────────────

def domain_age_memberships(age_days: int) -> Dict[str, float]:
    """
    very_low : 0–60 days   (brand-new domains → suspicious)
    low      : 30–180 days
    medium   : 120–730 days
    high     : 500+ days   (established domains → trustworthy)
    """
    if age_days < 0:  # unknown → treat as very_low
        age_days = 0
    return {
        "very_low": trapezoid(age_days, 0,  0,  30,  90),
        "low":      trapezoid(age_days, 30, 60, 120, 210),
        "medium":   trapezoid(age_days, 120,180, 365, 730),
        "high":     trapezoid(age_days, 500, 730, 9999, 99999),
    }


# ─────────────────────────────────────────────
#  SSL VALIDITY MEMBERSHIP
# ─────────────────────────────────────────────

def ssl_memberships(ssl_days: int) -> Dict[str, float]:
    """
    invalid : ≤ 0 days
    low     : 1–30 days   (expiring soon)
    valid   : 20–365 days
    strong  : 300+ days
    """
    if ssl_days < 0:
        ssl_days = 0
    return {
        "invalid": trapezoid(ssl_days, -1,  0,   0,   5),
        "low":     trapezoid(ssl_days,  1,  5,  20,  45),
        "valid":   trapezoid(ssl_days, 20, 45, 200, 365),
        "strong":  trapezoid(ssl_days, 300, 365, 9999, 99999),
    }


# ─────────────────────────────────────────────
#  CONFIDENCE MEMBERSHIP
# ─────────────────────────────────────────────

def confidence_membership(prob: float) -> Dict[str, float]:
    """Classify model probability into fuzzy confidence levels"""
    return {
        "low":    trapezoid(prob, 0.0, 0.0, 0.35, 0.55),
        "medium": trapezoid(prob, 0.4, 0.5, 0.65, 0.80),
        "high":   trapezoid(prob, 0.65, 0.80, 1.0, 1.0),
    }


# ─────────────────────────────────────────────
#  WEIGHT SCALE
# ─────────────────────────────────────────────

WEIGHT_SCALE = {
    "very_low":  0.10,
    "low":       0.20,
    "medium":    0.50,
    "high":      0.80,
    "very_high": 1.00,
}


# ─────────────────────────────────────────────
#  FUZZY RULE ENGINE
# ─────────────────────────────────────────────

@dataclass
class FuzzyContext:
    domain_age_days: int
    ssl_validity_days: int
    has_ip: int           # 1 or 0
    uses_https: int       # 1 or 0
    has_dns: int          # 1 or 0
    rf_pred: int          # 0=legit, 1=phish
    xgb_pred: int
    svm_pred: int
    rf_prob: float        # probability of phish
    xgb_prob: float
    svm_prob: float


def compute_fuzzy_weights(ctx: FuzzyContext) -> Dict[str, float]:
    """
    Apply fuzzy rules to compute dynamic weights for RF, XGB, SVM.
    Returns normalized weights summing to 1.0.
    """
    # Initial base weights (can be tuned by cross-validation)
    w_rf  = 0.33
    w_xgb = 0.34
    w_svm = 0.33

    age_mem = domain_age_memberships(ctx.domain_age_days)
    ssl_mem = ssl_memberships(ctx.ssl_validity_days)

    # ── RULE 1: Brand-new domain + RF predicts phish → boost RF
    if ctx.rf_pred == 1:
        activation = age_mem["very_low"]
        w_rf += 0.4 * activation

    # ── RULE 2: Valid SSL + SVM predicts legit → boost SVM
    if ctx.svm_pred == 0:
        activation = ssl_mem["valid"] + ssl_mem["strong"]
        w_svm += 0.3 * min(1.0, activation)

    # ── RULE 3: Strong SSL → XGB and SVM trusted more (less noisy signal)
    strong_ssl = ssl_mem["strong"]
    w_xgb += 0.2 * strong_ssl
    w_svm += 0.2 * strong_ssl

    # ── RULE 4: IP address in URL → RF is strongest judge
    if ctx.has_ip == 1:
        w_rf  += 0.5
        w_xgb += 0.1
        w_svm -= 0.1  # SVM less reliable for IP URLs

    # ── RULE 5: No DNS record → XGB penalized, RF trusted
    if ctx.has_dns == 0:
        w_rf  += 0.3
        w_xgb -= 0.1

    # ── RULE 6: No HTTPS → XGB and RF boosted, SVM penalized
    if ctx.uses_https == 0:
        w_xgb += 0.2
        w_rf  += 0.1
        w_svm -= 0.2

    # ── RULE 7: All three agree → amplify all
    if ctx.rf_pred == ctx.xgb_pred == ctx.svm_pred:
        w_rf  *= 1.2
        w_xgb *= 1.2
        w_svm *= 1.2

    # ── RULE 8: Old legitimate-looking domain but no SSL → reduce SVM
    if ctx.svm_pred == 0 and ssl_mem["invalid"] > 0.5:
        w_svm *= 0.5

    # Clamp all weights to [0.05, 2.0]
    w_rf  = max(0.05, w_rf)
    w_xgb = max(0.05, w_xgb)
    w_svm = max(0.05, w_svm)

    # Normalize to sum = 1
    total = w_rf + w_xgb + w_svm
    return {
        "RF":  round(w_rf  / total, 4),
        "XGB": round(w_xgb / total, 4),
        "SVM": round(w_svm / total, 4),
    }


def compute_fuzzy_confidence(ctx: FuzzyContext, weights: Dict[str, float]) -> float:
    """
    Weighted confidence score C using fuzzy-adjusted weights.
    P = Σ(Cᵢ × Wᵢ) / ΣWᵢ   (ΣWᵢ = 1 after normalization)
    """
    P = (
        ctx.rf_prob  * weights["RF"]  +
        ctx.xgb_prob * weights["XGB"] +
        ctx.svm_prob * weights["SVM"]
    )
    return round(P, 4)


def classify_result(P: float) -> str:
    if P > 0.8:
        return "PHISHING"
    elif P > 0.4:
        return "SUSPICIOUS"
    else:
        return "LEGITIMATE"


if __name__ == "__main__":
    # Example context
    ctx = FuzzyContext(
        domain_age_days=5,
        ssl_validity_days=0,
        has_ip=0,
        uses_https=0,
        has_dns=1,
        rf_pred=1, xgb_pred=1, svm_pred=1,
        rf_prob=0.92, xgb_prob=0.88, svm_prob=0.79,
    )
    weights = compute_fuzzy_weights(ctx)
    P = compute_fuzzy_confidence(ctx, weights)
    verdict = classify_result(P)

    print("\n=== Fuzzy Engine Output ===")
    print(f"  Weights  → RF: {weights['RF']}  XGB: {weights['XGB']}  SVM: {weights['SVM']}")
    print(f"  Score P  → {P}")
    print(f"  Verdict  → {verdict}")
