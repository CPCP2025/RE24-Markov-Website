"""
buckets.py
----------
Defines LA/EV buckets and computes per-bucket outcome probability distributions
from historical Statcast data.

Design
------
* Launch angle (LA) is bucketed in 10-degree bins from -90 to 90.
* Exit velocity (EV) is bucketed in 5 mph bins from 40 to 120 mph.
* Any ball outside those ranges is clipped to the nearest boundary bucket.
* Outcome probabilities are smoothed using Laplace (add-k) smoothing to
  prevent zero probabilities in sparse buckets.
* The BucketModel class can be serialised to/from a Parquet file so it
  doesn't need to be rebuilt every run.

Outcome mapping from Statcast 'events' column
---------------------------------------------
  single        -> Outcome.SINGLE
  double        -> Outcome.DOUBLE
  triple        -> Outcome.TRIPLE
  home_run      -> Outcome.HOME_RUN
  field_out, force_out, grounded_into_double_play,
  double_play, triple_play, fielders_choice,
  fielders_choice_out, other_out  -> Outcome.FIELD_OUT
  strikeout, strikeout_double_play -> Outcome.STRIKEOUT
  walk, intent_walk                -> Outcome.WALK
  hit_by_pitch                    -> Outcome.HBP
  sac_fly, sac_bunt, sac_fly_double_play -> Outcome.SAC
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import Bucket, Outcome

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data"
BUCKET_MODEL_CACHE = CACHE_DIR / "bucket_model.pkl"

# ---------------------------------------------------------------------------
# Statcast event -> Outcome mapping
# ---------------------------------------------------------------------------

_EVENT_MAP: Dict[str, Outcome] = {
    "single": Outcome.SINGLE,
    "double": Outcome.DOUBLE,
    "triple": Outcome.TRIPLE,
    "home_run": Outcome.HOME_RUN,
    "field_out": Outcome.FIELD_OUT,
    "force_out": Outcome.FIELD_OUT,
    "grounded_into_double_play": Outcome.FIELD_OUT,
    "double_play": Outcome.FIELD_OUT,
    "triple_play": Outcome.FIELD_OUT,
    "fielders_choice": Outcome.FIELD_OUT,
    "fielders_choice_out": Outcome.FIELD_OUT,
    "other_out": Outcome.FIELD_OUT,
    "strikeout": Outcome.STRIKEOUT,
    "strikeout_double_play": Outcome.STRIKEOUT,
    "walk": Outcome.WALK,
    "intent_walk": Outcome.WALK,
    "hit_by_pitch": Outcome.HBP,
    "sac_fly": Outcome.SAC,
    "sac_bunt": Outcome.SAC,
    "sac_fly_double_play": Outcome.SAC,
    "catcher_interf": Outcome.WALK,  # treat as walk equivalent
}

ALL_OUTCOMES: List[Outcome] = list(Outcome)


def map_event(event: str) -> Optional[Outcome]:
    """Map a raw Statcast event string to an Outcome enum."""
    if pd.isna(event):
        return None
    return _EVENT_MAP.get(str(event).strip().lower(), None)


# ---------------------------------------------------------------------------
# Bucket grid definition
# ---------------------------------------------------------------------------

LA_EDGES: List[float] = list(range(-90, 93, 3))   # -90, -87, ..., 90  (61 edges, 60 bins)
EV_EDGES: List[float] = list(range(40, 127, 3))   # 40, 43, ..., 124  (29 edges, 28 bins ... up to 127)


def _build_bucket_grid() -> List[Bucket]:
    """Return every (LA bin, EV bin) combination as a flat list."""
    buckets = []
    for i in range(len(LA_EDGES) - 1):
        for j in range(len(EV_EDGES) - 1):
            buckets.append(Bucket(
                la_lo=LA_EDGES[i], la_hi=LA_EDGES[i + 1],
                ev_lo=EV_EDGES[j], ev_hi=EV_EDGES[j + 1],
            ))
    return buckets


ALL_BUCKETS: List[Bucket] = _build_bucket_grid()


def assign_bucket(la: float, ev: float) -> Bucket:
    """
    Assign a (LA, EV) pair to a Bucket, clipping to boundary bins if out of range.
    """
    la_clipped = float(np.clip(la, LA_EDGES[0], LA_EDGES[-1] - 1e-9))
    ev_clipped = float(np.clip(ev, EV_EDGES[0], EV_EDGES[-1] - 1e-9))

    la_idx = int(np.searchsorted(LA_EDGES, la_clipped, side="right") - 1)
    la_idx = max(0, min(la_idx, len(LA_EDGES) - 2))

    ev_idx = int(np.searchsorted(EV_EDGES, ev_clipped, side="right") - 1)
    ev_idx = max(0, min(ev_idx, len(EV_EDGES) - 2))

    return Bucket(
        la_lo=LA_EDGES[la_idx], la_hi=LA_EDGES[la_idx + 1],
        ev_lo=EV_EDGES[ev_idx], ev_hi=EV_EDGES[ev_idx + 1],
    )


# ---------------------------------------------------------------------------
# BucketModel: stores per-bucket outcome probability tables
# ---------------------------------------------------------------------------

class BucketModel:
    """
    Stores P(outcome | bucket) for every (LA, EV) bucket.

    Parameters
    ----------
    smoothing_k : Laplace smoothing pseudo-count added to each outcome cell.
                  Higher k → more shrinkage toward uniform; 1.0 is a reasonable default.
    """

    def __init__(self, smoothing_k: float = 1.0):
        self.smoothing_k = smoothing_k
        # bucket -> {Outcome -> probability}
        self._probs: Dict[Bucket, Dict[Outcome, float]] = {}
        # bucket -> raw count of batted balls
        self._counts: Dict[Bucket, int] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "BucketModel":
        """
        Fit the model from a DataFrame of plate-appearance events.

        Expects columns: launch_angle, launch_speed, events
        Only rows with valid LA and EV (batted-ball events) are used.
        Non-batted-ball events (K, BB, HBP) are noted but not assigned
        to buckets – those rates are modelled separately in PlayerProfile.
        """
        batted = df[df["launch_angle"].notna() & df["launch_speed"].notna()].copy()
        batted["_outcome"] = batted["events"].map(map_event)
        batted = batted[batted["_outcome"].notna()]

        # Assign buckets
        batted["_bucket"] = batted.apply(
            lambda r: assign_bucket(r["launch_angle"], r["launch_speed"]), axis=1
        )

        # Count outcomes per bucket
        counts: Dict[Bucket, Dict[Outcome, int]] = {b: {o: 0 for o in ALL_OUTCOMES} for b in ALL_BUCKETS}
        bucket_totals: Dict[Bucket, int] = {b: 0 for b in ALL_BUCKETS}

        for _, row in batted.iterrows():
            bkt = row["_bucket"]
            outcome = row["_outcome"]
            counts[bkt][outcome] += 1
            bucket_totals[bkt] += 1

        # Convert to probabilities with Laplace smoothing
        k = self.smoothing_k
        n_outcomes = len(ALL_OUTCOMES)
        for bkt in ALL_BUCKETS:
            total = bucket_totals[bkt] + k * n_outcomes
            self._probs[bkt] = {
                o: (counts[bkt][o] + k) / total for o in ALL_OUTCOMES
            }
            self._counts[bkt] = bucket_totals[bkt]

        self._is_fitted = True
        logger.info(
            f"BucketModel fitted: {len(batted):,} batted balls across "
            f"{sum(1 for b in ALL_BUCKETS if self._counts[b] > 0)} non-empty buckets"
        )
        return self

    def predict(self, la: float, ev: float) -> Dict[Outcome, float]:
        """
        Return a probability distribution over Outcomes for a given LA/EV.
        """
        if not self._is_fitted:
            raise RuntimeError("BucketModel must be fitted before calling predict()")
        bkt = assign_bucket(la, ev)
        return self._probs[bkt].copy()

    def predict_from_bucket(self, bucket: Bucket) -> Dict[Outcome, float]:
        """Return outcome probabilities for a pre-assigned bucket."""
        if not self._is_fitted:
            raise RuntimeError("BucketModel must be fitted before calling predict()")
        return self._probs[bucket].copy()

    def bucket_count(self, bucket: Bucket) -> int:
        """Number of observed batted balls in this bucket."""
        return self._counts.get(bucket, 0)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert probability table to a tidy DataFrame for inspection."""
        rows = []
        for bkt, probs in self._probs.items():
            for outcome, prob in probs.items():
                rows.append({
                    "la_lo": bkt.la_lo,
                    "la_hi": bkt.la_hi,
                    "ev_lo": bkt.ev_lo,
                    "ev_hi": bkt.ev_hi,
                    "la_mid": bkt.la_mid,
                    "ev_mid": bkt.ev_mid,
                    "outcome": outcome.value,
                    "probability": prob,
                    "n_batted_balls": self._counts.get(bkt, 0),
                })
        return pd.DataFrame(rows)

    def save(self, path: Path = BUCKET_MODEL_CACHE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"BucketModel saved to {path}")

    @classmethod
    def load(cls, path: Path = BUCKET_MODEL_CACHE) -> "BucketModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"BucketModel loaded from {path}")
        return model

    @classmethod
    def load_or_fit(
        cls,
        df: Optional[pd.DataFrame] = None,
        smoothing_k: float = 1.0,
        force_refit: bool = False,
    ) -> "BucketModel":
        """
        Convenience factory: load cached model if available, else fit from df.
        """
        if BUCKET_MODEL_CACHE.exists() and not force_refit:
            return cls.load()
        if df is None:
            raise ValueError("Must provide df when no cached model exists")
        model = cls(smoothing_k=smoothing_k).fit(df)
        model.save()
        return model


# ---------------------------------------------------------------------------
# Draft / validation helpers
# ---------------------------------------------------------------------------

def get_observed_outcome(row: pd.Series) -> Optional[Outcome]:
    """
    Return the observed Outcome from a Statcast row.
    Used in validation mode to compare against reconstructed predictions.
    """
    return map_event(row.get("events", None))


def validation_report(df: pd.DataFrame, model: BucketModel, n_sample: int = 1000) -> pd.DataFrame:
    """
    Draft validation: compare observed outcomes vs most-probable bucket prediction
    for a random sample of batted-ball events.

    Returns a DataFrame with columns:
        player_name, launch_angle, launch_speed, bucket,
        observed, predicted_top, predicted_prob, correct
    """
    batted = df[df["launch_angle"].notna() & df["launch_speed"].notna()].copy()
    batted["_outcome_obs"] = batted["events"].map(map_event)
    batted = batted[batted["_outcome_obs"].notna()]

    sample = batted.sample(min(n_sample, len(batted)), random_state=42)

    rows = []
    for _, row in sample.iterrows():
        la = row["launch_angle"]
        ev = row["launch_speed"]
        bkt = assign_bucket(la, ev)
        probs = model.predict(la, ev)
        pred_top = max(probs, key=probs.get)
        obs = row["_outcome_obs"]
        rows.append({
            "player_name": row.get("player_name", ""),
            "launch_angle": la,
            "launch_speed": ev,
            "bucket": str(bkt),
            "observed": obs.value if obs else None,
            "predicted_top": pred_top.value,
            "predicted_prob": round(probs[pred_top], 3),
            "correct": obs == pred_top,
            "raw_event": row.get("events", ""),
        })

    report_df = pd.DataFrame(rows)
    accuracy = report_df["correct"].mean()
    logger.info(f"Validation accuracy (top-1): {accuracy:.1%} on {len(report_df)} samples")
    return report_df
