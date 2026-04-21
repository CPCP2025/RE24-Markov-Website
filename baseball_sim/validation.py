"""
validation.py
-------------
Draft validation mode: compare observed Statcast outcomes against the bucket
model's predictions to verify that bucket assignment and event parsing are
working correctly.

Usage
-----
    from baseball_sim.validation import run_validation
    report = run_validation(df, bucket_model, n_sample=500)
    print(report.to_string())

The report shows:
  * observed outcome
  * predicted top-1 outcome (argmax of bucket probability)
  * probability assigned to the predicted outcome
  * whether the prediction matched
  * summary accuracy by outcome type
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from .buckets import BucketModel, assign_bucket, map_event, get_observed_outcome
from .models import Outcome

logger = logging.getLogger(__name__)


def run_validation(
    df: pd.DataFrame,
    model: BucketModel,
    n_sample: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Validate bucket assignments and outcome probabilities on a sample of events.

    Parameters
    ----------
    df           : DataFrame of Statcast events (must have launch_angle,
                   launch_speed, events columns)
    model        : fitted BucketModel
    n_sample     : number of batted-ball events to sample
    random_state : for reproducibility

    Returns
    -------
    DataFrame with one row per event showing observed vs reconstructed outcome.
    """
    batted = df[df["launch_angle"].notna() & df["launch_speed"].notna()].copy()
    batted["_obs_outcome"] = batted["events"].apply(
        lambda e: map_event(e).value if map_event(e) else None
    )
    batted = batted[batted["_obs_outcome"].notna()]

    n = min(n_sample, len(batted))
    sample = batted.sample(n, random_state=random_state)

    rows = []
    for _, row in sample.iterrows():
        la = float(row["launch_angle"])
        ev = float(row["launch_speed"])
        bkt = assign_bucket(la, ev)
        probs = model.predict(la, ev)
        pred_top = max(probs, key=probs.get)
        obs_str = row["_obs_outcome"]
        obs_outcome = Outcome(obs_str) if obs_str else None
        pred_prob = probs[pred_top]
        obs_prob = probs.get(obs_outcome, 0.0) if obs_outcome else 0.0

        rows.append({
            "player_name": row.get("player_name", ""),
            "game_date": str(row.get("game_date", ""))[:10],
            "launch_angle": round(la, 1),
            "launch_speed": round(ev, 1),
            "bucket_la": f"{bkt.la_lo:.0f}–{bkt.la_hi:.0f}",
            "bucket_ev": f"{bkt.ev_lo:.0f}–{bkt.ev_hi:.0f}",
            "bucket_n": model.bucket_count(bkt),
            "observed": obs_str,
            "predicted_top": pred_top.value,
            "predicted_prob": round(pred_prob, 3),
            "observed_prob": round(obs_prob, 3),
            "correct": obs_outcome == pred_top,
            "raw_event": row.get("events", ""),
            "bb_type": row.get("bb_type", ""),
        })

    report = pd.DataFrame(rows)
    _log_summary(report)
    return report


def _log_summary(report: pd.DataFrame) -> None:
    """Log a per-outcome accuracy breakdown."""
    overall = report["correct"].mean()
    logger.info(f"\n{'='*50}")
    logger.info(f"VALIDATION SUMMARY  ({len(report)} samples)")
    logger.info(f"Overall top-1 accuracy: {overall:.1%}")
    logger.info(f"{'='*50}")

    for obs in sorted(report["observed"].unique()):
        subset = report[report["observed"] == obs]
        acc = subset["correct"].mean()
        logger.info(f"  {obs:20s}: {acc:.1%}  (n={len(subset)})")


def outcome_confusion_matrix(report: pd.DataFrame) -> pd.DataFrame:
    """
    Build a confusion matrix from the validation report.
    Rows = observed outcomes, Columns = predicted top-1 outcomes.
    """
    outcomes = sorted(
        set(report["observed"].tolist() + report["predicted_top"].tolist())
    )
    cm = pd.crosstab(
        report["observed"],
        report["predicted_top"],
        rownames=["Observed"],
        colnames=["Predicted"],
    )
    return cm


def accuracy_by_bucket_size(report: pd.DataFrame) -> pd.DataFrame:
    """
    Bin events by their bucket sample size and compute accuracy within each bin.
    Useful for understanding how sparsity affects prediction quality.
    """
    bins = [0, 10, 50, 200, 1000, 999_999]
    labels = ["<10", "10–50", "50–200", "200–1000", ">1000"]
    report = report.copy()
    report["bucket_size_bin"] = pd.cut(
        report["bucket_n"], bins=bins, labels=labels, right=True
    )
    return (
        report.groupby("bucket_size_bin", observed=True)["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_events"})
        .reset_index()
    )
