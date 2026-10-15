"""Summary-result screening for the controlled resampling-before-splitting signature.
The score is not a probability of leakage in a published study.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence
import math

import numpy as np

from ._forest import Forest, load_forest, export_sklearn

__version__ = "1.0.1"
__all__ = ["check", "check_many", "Result", "MODEL", "model_info", "selftest",
           "Forest", "load_forest", "export_sklearn"]

MODEL: Forest = load_forest()

# Recognised spellings for the seven fitted families.
_FAMILY = {
    "logreg": "LogReg", "logisticregression": "LogReg", "logistic": "LogReg",
    "lr": "LogReg",

    "dectree": "DecTree", "decisiontree": "DecTree", "cart": "DecTree",
    "tree": "DecTree", "j48": "DecTree",
    "5nn": "5NN", "knn": "5NN", "kneighbors": "5NN", "nearestneighbors": "5NN",
    "svm-rbf": "SVM-RBF", "svm": "SVM-RBF", "svc": "SVM-RBF", "rbf": "SVM-RBF",
    "randforest": "RandForest", "randomforest": "RandForest", "rf": "RandForest",

    "gradboost": "GradBoost", "gradientboosting": "GradBoost", "gbm": "GradBoost",

    "xgboost": "XGBoost", "xgb": "XGBoost",
}

THRESHOLDS = {"low screening score": 0.35, "intermediate screening score": 0.65, "high screening score - methods check recommended": 1.01}


@dataclass
class Result:
    """The outcome of one screen."""
    score: float
    verdict: str
    f1: float
    imbalance_ratio: float
    n_minority: int
    n_features: int
    clf: str
    family: str
    expected_honest_f1: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def probability(self) -> float:
        """Legacy numeric alias; this value is a screening score, not calibrated risk."""
        return self.score

    def __repr__(self) -> str:
        return (f"Result(score={self.score:.3f}, "
                f"verdict={self.verdict!r}, f1={self.f1}, "
                f"imbalance_ratio={self.imbalance_ratio})")

    def explain(self) -> str:
        """A paragraph a reviewer can paste into a report."""
        lines = [
            f"Reported minority-class F1 of {self.f1:.3f} on a problem with "
            f"imbalance ratio {self.imbalance_ratio:g}, {self.n_minority} minority "
            f"cases and {self.n_features} features, using {self.clf} "
            f"(screened as the {self.family} family).",
            f"Screening score: {self.score:.2f} -- {self.verdict}. "
            "This score is not a calibrated probability that a study contains leakage.",
        ]
        if self.expected_honest_f1 is not None:
            lines.append(
                f"On this corpus, correctly evaluated results at that difficulty "
                f"average about F1 {self.expected_honest_f1:.2f}.")
        lines += self.notes
        lines.append("This is a screen, not a finding. The appropriate response is to "
                     "ask where the resampler sits relative to the split.")
        return " ".join(lines)


# ---------------------------------------------------------------------------
def _family(clf: str | None) -> str:
    if not clf:
        raise ValueError("Classifier family is required; choose a supported family explicitly.")
    key = str(clf).strip().lower().replace(" ", "").replace("_", "")
    family = _FAMILY.get(key)
    if family is None:
        raise ValueError(f"Unsupported classifier family {clf!r}; supported: "
                         + ", ".join(MODEL.metadata["classifiers"]))
    return family


def _row(f1: float, imbalance_ratio: float, n_minority: int, n_features: int,
         family: str) -> np.ndarray:
    x = {c: 0.0 for c in MODEL.columns}
    x["F1"] = float(f1)
    x["log_ir"] = math.log10(float(imbalance_ratio))
    x["log_nmin"] = math.log10(float(n_minority))
    x["log_p"] = math.log10(float(n_features))
    key = f"clf_{family}"
    if key in x:
        x[key] = 1.0
    return np.array([x[c] for c in MODEL.columns], dtype=float)


def _validate(f1, imbalance_ratio, n_minority, n_features) -> list[str]:
    notes = []
    if not all(math.isfinite(float(v)) for v in (f1, imbalance_ratio, n_minority, n_features)):
        raise ValueError("All numeric inputs must be finite")
    if n_minority != int(n_minority) or n_features != int(n_features):
        raise ValueError("Counts must be integers")
    if not 0.0 <= f1 <= 1.0:
        raise ValueError("f1 must be between 0 and 1")
    if imbalance_ratio < 1:
        raise ValueError("imbalance_ratio must be at least 1")
    if n_minority < 1 or n_features < 1:
        raise ValueError("n_minority and n_features must be at least 1")
    m = MODEL.metadata
    lo, hi = m.get("ir_range", (1.0, 110.0))
    if not lo <= imbalance_ratio <= hi:
        notes.append(f"Imbalance ratio {imbalance_ratio:g} is outside the "
                     f"{lo:g}-{hi:g} range the model was fitted on, so the "
                     f"screening score is an extrapolation.")
    lo, hi = m.get("nmin_range", (30, 2000))
    if not lo <= n_minority <= hi:
        notes.append(f"Minority count {n_minority} is outside the fitted "
                     f"{lo:g}-{hi:g} range.")
    lo, hi = m.get("p_range", (3, 180))
    if not lo <= n_features <= hi:
        notes.append(f"Feature count {n_features} is outside the fitted {lo:g}-{hi:g} range.")
    return notes


def _verdict(p: float) -> str:
    for label, cut in THRESHOLDS.items():
        if p < cut:
            return label
    return "high screening score - methods check recommended"


def check(f1: float, imbalance_ratio: float, n_minority: int, n_features: int,
          clf: str | None = None) -> Result:
    """Screen a single reported result.

    Parameters
    ----------
    f1
        The reported minority-class F1.
    imbalance_ratio
        Majority cases divided by minority cases in the dataset as analysed.
    n_minority
        Number of minority cases before any resampling.
    n_features
        Number of predictors entering the model, after encoding.
    clf
        Supported classifier family or recognised spelling. Unsupported or missing
        families raise ValueError; approximate cross-family mappings are refused.

    Returns
    -------
    Result
    """
    notes = _validate(f1, imbalance_ratio, n_minority, n_features)
    fam = _family(clf)
    x = _row(f1, imbalance_ratio, n_minority, n_features, fam)
    p = float(MODEL.predict_proba(x[None, :])[0])
    exp = MODEL.metadata.get("honest_curve")
    e = None
    if exp:
        li = math.log10(float(imbalance_ratio))
        e = float(np.clip(exp["intercept"] + exp["slope"] * li, 0.0, 1.0))
    return Result(score=p, verdict=_verdict(p), f1=float(f1),
                  imbalance_ratio=float(imbalance_ratio), n_minority=int(n_minority),
                  n_features=int(n_features), clf=str(clf or "unspecified"),
                  family=fam, expected_honest_f1=e, notes=notes)


def check_many(records: Iterable[dict] | "Sequence[dict]"):
    """Screen many results at once.

    Accepts an iterable of dicts with the same keys as :func:`check`.  Returns a
    pandas DataFrame if pandas is importable, otherwise a list of Result.
    """
    out = [check(**r) for r in records]
    try:
        import pandas as pd
    except ImportError:
        return out
    return pd.DataFrame([{
        "f1": r.f1, "imbalance_ratio": r.imbalance_ratio, "n_minority": r.n_minority,
        "n_features": r.n_features, "clf": r.clf, "family": r.family,
        "score": r.score, "verdict": r.verdict,
        "expected_honest_f1": r.expected_honest_f1} for r in out])


def model_info() -> dict:
    """Provenance of the shipped screening model."""
    d = dict(MODEL.metadata)
    d["n_trees"] = len(MODEL.trees)
    d["features"] = list(MODEL.columns)
    d["version"] = __version__
    return d


def selftest() -> bool:
    """Check the shipped model against its stored regression fixtures."""
    fx = MODEL.metadata.get("fixtures", [])
    if not fx:
        raise RuntimeError("this model.json carries no fixtures")
    ok = True
    for row in fx:
        got = check(**{k: row[k] for k in
                       ("f1", "imbalance_ratio", "n_minority", "n_features", "clf")}).probability
        if abs(got - row["probability"]) > 1e-6:
            ok = False
            print(f"MISMATCH {row} -> {got:.8f}")
    return ok
