"""Train & compare the Coverage Clarity / Risk scoring models.

Usage: python scripts/train_model.py [--data path/to.csv]
Outputs: models/coverage_risk_model.joblib, models/model_metrics.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coverwise import config  # noqa: E402
from coverwise.scoring import train_models  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.SYNTHETIC_DATASET))
    args = ap.parse_args()
    m = train_models(Path(args.data))
    print("\n=== Model comparison (5-fold CV on train split, macro-F1) ===")
    for name, r in m["model_comparison"].items():
        print(f"{name:22s} CV F1 {r['cv_macro_f1_mean']:.3f} +/- {r['cv_macro_f1_std']:.3f} | "
              f"test acc {r['test_accuracy']:.3f} | test F1 {r['test_macro_f1']:.3f}")
    print(f"\nBest model: {m['best_model']}")
    print("Rule-based baseline:", m["rule_baseline"])
    print("Score regressor:", m["score_regressor"])
    rep = m["best_model_test_report"]
    print("\nPer-class (best model, held-out test):")
    for lbl in config.RISK_LABELS:
        r = rep.get(lbl, {})
        print(f"  {lbl:18s} P {r.get('precision', 0):.2f} R {r.get('recall', 0):.2f} F1 {r.get('f1-score', 0):.2f}")
    print("Confusion matrix (rows=true, cols=pred):", json.dumps(m["confusion_matrix"]))
