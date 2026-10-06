"""Coverage Clarity / Risk Scoring.

1. Transparent rule-based baseline
   Weights come from a linear regression of coverage_clarity_score on the
   features of the synthetic dataset (R^2 = 0.96), rounded to readable points.
   Every point added/removed is shown to the user as a "reason".
2. ML models (Logistic Regression, Decision Tree, Random Forest, Gradient
   Boosting) trained on the same features - best model chosen by macro-F1.
   A Gradient Boosting regressor predicts the 0-100 Coverage Clarity Score.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)
MODEL_PATH = config.MODELS_DIR / "coverage_risk_model.joblib"
METRICS_PATH = config.MODELS_DIR / "model_metrics.json"

# points per unit of feature (derived from regression, see notebooks/01)
RULE_BASE = 103.0
RULE_WEIGHTS = {
    "risk_flags_count": (-7.0, "each risk flag"),
    "copayment_percentage": (-0.75, "per % co-payment"),
    "pre_existing_disease_waiting_period_months": (-0.25, "per month of PED waiting"),
    "ambiguous_clause_count": (-2.0, "per vague clause"),
    "restoration_benefit_present": (10.0, "restoration benefit available"),
    "no_claim_bonus_present": (5.0, "no-claim / cumulative bonus available"),
    "maternity_coverage_present": (1.0, "maternity covered"),
    "child_coverage_present": (0.5, "child cover available"),
}
LABEL_THRESHOLDS = [(80, "Strong Coverage"), (58, "Moderate Coverage"), (38, "Caution Required"),
                    (0, "High Risk")]


def label_for_score(score: float) -> str:
    for th, label in LABEL_THRESHOLDS:
        if score >= th:
            return label
    return "High Risk"


def rule_based_score(f: dict) -> dict:
    contributions = []
    score = RULE_BASE
    for feat, (w, desc) in RULE_WEIGHTS.items():
        pts = w * float(f.get(feat, 0))
        if abs(pts) >= 0.5:
            contributions.append({"feature": feat, "value": f.get(feat), "points": round(pts, 1),
                                  "description": desc})
        score += pts
    score = float(np.clip(score, 0, 100))
    contributions.sort(key=lambda c: abs(c["points"]), reverse=True)
    return {"score": round(score), "label": label_for_score(score), "contributions": contributions}


# ------------------------------------------------------------------- training
def _frame(features: dict | list[dict]) -> pd.DataFrame:
    rows = features if isinstance(features, list) else [features]
    df = pd.DataFrame(rows)
    return df[config.CATEGORICAL_FEATURES + config.MODEL_FEATURES]


def train_models(csv_path: Path = config.SYNTHETIC_DATASET, out_path: Path = MODEL_PATH,
                 random_state: int = 42) -> dict:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 f1_score, mean_absolute_error, r2_score)
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    df = pd.read_csv(csv_path)
    X = df[config.CATEGORICAL_FEATURES + config.MODEL_FEATURES]
    y = df["risk_label"]
    y_score = df["coverage_clarity_score"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", categories=[config.POLICY_TYPES]),
         config.CATEGORICAL_FEATURES),
        ("num", StandardScaler(), config.MODEL_FEATURES),
    ])
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, y_score, test_size=0.2, stratify=y, random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    comparison = {}
    for name, est in candidates.items():
        pipe = Pipeline([("pre", pre), ("clf", est)])
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        comparison[name] = {
            "cv_macro_f1_mean": round(float(cv_f1.mean()), 4),
            "cv_macro_f1_std": round(float(cv_f1.std()), 4),
            "test_accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "test_macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        }
        log.info("%s: %s", name, comparison[name])

    best_name = max(comparison, key=lambda n: comparison[n]["cv_macro_f1_mean"])
    best = Pipeline([("pre", pre), ("clf", candidates[best_name])]).fit(X_tr, y_tr)
    pred = best.predict(X_te)
    report = classification_report(y_te, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=config.RISK_LABELS).tolist()

    reg = Pipeline([("pre", pre), ("reg", GradientBoostingRegressor(random_state=random_state))])
    reg.fit(X_tr, s_tr)
    s_pred = reg.predict(X_te)

    # rule baseline on the same test split
    rule_pred = [rule_based_score(r)["label"] for r in X_te.to_dict("records")]
    rule_scores = [rule_based_score(r)["score"] for r in X_te.to_dict("records")]

    metrics = {
        "best_model": best_name,
        "model_comparison": comparison,
        "best_model_test_report": report,
        "confusion_matrix_labels": config.RISK_LABELS,
        "confusion_matrix": cm,
        "score_regressor": {"test_mae": round(float(mean_absolute_error(s_te, s_pred)), 2),
                            "test_r2": round(float(r2_score(s_te, s_pred)), 4)},
        "rule_baseline": {"test_accuracy": round(float(accuracy_score(y_te, rule_pred)), 4),
                          "test_macro_f1": round(float(f1_score(y_te, rule_pred, average="macro")), 4),
                          "score_mae": round(float(mean_absolute_error(s_te, rule_scores)), 2)},
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    # refit on all data for serving
    best.fit(X, y)
    reg.fit(X, y_score)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": best, "regressor": reg, "model_name": best_name,
                 "features": config.MODEL_FEATURES}, out_path)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


# ------------------------------------------------------------------- inference
_BUNDLE = None


def load_model(auto_train: bool = True):
    """Load the trained bundle; (re)train automatically if missing or saved with
    an incompatible scikit-learn version (training takes a few seconds)."""
    global _BUNDLE
    if _BUNDLE is None:
        import joblib
        try:
            _BUNDLE = joblib.load(MODEL_PATH)
        except Exception as exc:
            log.warning("Model not loadable (%s)", exc)
            _BUNDLE = False
            if auto_train and config.SYNTHETIC_DATASET.exists():
                try:
                    train_models()
                    _BUNDLE = joblib.load(MODEL_PATH)
                except Exception as exc2:  # fall back to rules
                    log.warning("Auto-training failed (%s); using rule-based scoring", exc2)
    return _BUNDLE or None


def score_policy(features: dict) -> dict:
    rule = rule_based_score(features)
    out = {"rule_score": rule["score"], "rule_label": rule["label"],
           "reasons": rule["contributions"], "method": "rule-based"}
    bundle = load_model()
    if bundle:
        X = _frame(features)
        proba = bundle["classifier"].predict_proba(X)[0]
        classes = list(bundle["classifier"].classes_)
        label = classes[int(np.argmax(proba))]
        ml_score = float(np.clip(bundle["regressor"].predict(X)[0], 0, 100))
        out.update(
            method=f"ML ({bundle['model_name']}) + rule-based explanation",
            score=round(ml_score), label=label, confidence=round(float(max(proba)), 3),
            probabilities={c: round(float(p), 3) for c, p in zip(classes, proba)},
            agreement=label == rule["label"],
        )
    else:
        out.update(score=rule["score"], label=rule["label"], confidence=None,
                   probabilities={}, agreement=True)
    out["label_display"] = config.LABEL_DISPLAY.get(out["label"], out["label"])
    return out
