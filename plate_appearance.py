"""
plate_appearance.py
-------------------
Simulates a single plate appearance for a given PlayerProfile.

Algorithm
---------
For non-fallback players:
  1. Sample from non-contact outcomes first (K, BB, HBP, SAC) using
     the player's historical rates.  These are sequential Bernoulli trials
     whose cumulative probability is checked against a single uniform draw.
  2. If none triggered, it's a batted ball:
     a. Sample a bucket from the player's bucket_weights distribution.
     b. Retrieve outcome probabilities from the BucketModel at that bucket's midpoint.
     c. Sample the final outcome (single/double/triple/HR/field_out) from those probs.

For fallback players (is_fallback=True):
  Same step 1, but in step 2 use the flat fallback contact distribution
  instead of the BucketModel, and skip the bucket sampling.

Notes
-----
* The sequential Bernoulli approach slightly over-counts contact because
  (k_rate + bb_rate + hbp_rate + sac_rate) > true non-contact rate in edge cases.
  We normalise so total probability of each branch sums to 1.0.
* A NumPy RNG is injected for reproducibility.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .buckets import BucketModel
from .fallback import get_fallback_contact_probs
from .models import Bucket, Outcome, PAEvent, PlayerProfile

logger = logging.getLogger(__name__)


def simulate_pa(
    profile: PlayerProfile,
    bucket_model: BucketModel,
    rng: np.random.Generator,
) -> PAEvent:
    """
    Simulate one plate appearance.

    Parameters
    ----------
    profile      : PlayerProfile for the current batter
    bucket_model : fitted BucketModel (ignored for fallback players)
    rng          : NumPy random Generator for reproducibility

    Returns
    -------
    PAEvent with the simulated outcome
    """
    # ---- Step 1: non-contact outcome? ------------------------------------
    # Use pre-computed probability array from PlayerProfile.__post_init__
    # order: [K, BB, HBP, SAC, contact]
    outcomes_nc = [Outcome.STRIKEOUT, Outcome.WALK, Outcome.HBP, Outcome.SAC]
    choice_idx = int(rng.choice(5, p=profile._pa_probs))

    if choice_idx < len(outcomes_nc):
        # Non-contact outcome
        outcome = outcomes_nc[choice_idx]
        return PAEvent(
            player_id=profile.player_id,
            player_name=profile.player_name,
            outcome=outcome,
            launch_angle=None,
            exit_velocity=None,
            bucket=None,
        )

    # ---- Step 2: batted ball ---------------------------------------------
    if profile.is_fallback:
        return _simulate_fallback_contact(profile, rng)
    else:
        return _simulate_bucket_contact(profile, bucket_model, rng)


# Module-level cache: bucket -> (outcome_list, prob_array)
# Built once per unique bucket on first access, reused for every subsequent PA.
_bucket_outcome_cache: dict = {}


def _get_bucket_outcome_arrays(bucket_model: BucketModel, bkt: Bucket):
    """Return cached (outcomes, probs) arrays for this bucket."""
    key = (id(bucket_model), bkt)
    if key not in _bucket_outcome_cache:
        outcome_probs = bucket_model.predict_from_bucket(bkt)
        outcome_list = list(outcome_probs.keys())
        prob_arr = np.array([outcome_probs[o] for o in outcome_list], dtype=np.float64)
        prob_arr /= prob_arr.sum()
        _bucket_outcome_cache[key] = (outcome_list, prob_arr)
    return _bucket_outcome_cache[key]


def _simulate_bucket_contact(
    profile: PlayerProfile,
    bucket_model: BucketModel,
    rng: np.random.Generator,
) -> PAEvent:
    """Sample a bucket using pre-computed arrays, then sample outcome."""
    # Use pre-computed arrays from __post_init__ — no list/array rebuild per PA
    bkt_idx = int(rng.choice(len(profile._bucket_list), p=profile._bucket_weights_arr))
    bkt: Bucket = profile._bucket_list[bkt_idx]

    outcome_list, prob_arr = _get_bucket_outcome_arrays(bucket_model, bkt)
    outcome_idx = int(rng.choice(len(outcome_list), p=prob_arr))
    outcome: Outcome = outcome_list[outcome_idx]

    return PAEvent(
        player_id=profile.player_id,
        player_name=profile.player_name,
        outcome=outcome,
        launch_angle=bkt.la_mid,
        exit_velocity=bkt.ev_mid,
        bucket=bkt,
    )


def _simulate_fallback_contact(
    profile: PlayerProfile,
    rng: np.random.Generator,
) -> PAEvent:
    """Use the fallback contact distribution (MiLB-derived or prior)."""
    contact_probs = get_fallback_contact_probs(profile)
    outcomes = list(contact_probs.keys())
    probs = np.array([contact_probs[o] for o in outcomes], dtype=float)
    probs /= probs.sum()

    idx = int(rng.choice(len(outcomes), p=probs))
    outcome = outcomes[idx]

    return PAEvent(
        player_id=profile.player_id,
        player_name=profile.player_name,
        outcome=outcome,
        launch_angle=None,
        exit_velocity=None,
        bucket=None,
    )
