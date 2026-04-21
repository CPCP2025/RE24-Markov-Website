"""
player_profile.py
-----------------
Builds PlayerProfile objects from Statcast data.

A profile contains:
  * bucket_weights : fraction of the player's batted balls in each LA/EV bucket
  * k_rate, bb_rate, hbp_rate, sac_rate : non-contact outcome rates (per PA)

These non-contact rates are sampled *first* during simulation:
  1. Roll for K  → if hit, outcome = STRIKEOUT
  2. Roll for BB → if hit, outcome = WALK
  3. Roll for HBP → if hit, outcome = HBP
  4. Roll for SAC → if hit, outcome = SAC
  5. Otherwise → batted ball; pick bucket from bucket_weights, then outcome from BucketModel

This ordering is an assumption. In reality the rates overlap; we normalise
so the total non-contact probability is consistent with observed plate discipline.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .buckets import ALL_BUCKETS, assign_bucket, BucketModel
from .ingestion import filter_plate_appearances, load_batter_seasons
from .models import Bucket, Outcome, PlayerProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service-time estimation
# ---------------------------------------------------------------------------

def estimate_mlb_seasons(batter_id: int, all_available_seasons: List[int]) -> int:
    """
    Estimate how many MLB seasons a batter has by counting seasons with >=10 PA
    in Statcast data.  Real service time is not publicly available, so this is
    a proxy.
    """
    count = 0
    for season in all_available_seasons:
        df = load_batter_seasons(batter_id, [season])
        pa = filter_plate_appearances(df)
        if len(pa) >= 10:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

def build_player_profile(
    batter_id: int,
    player_name: str,
    seasons: List[int],
    service_time_threshold: float = 1.0,
    min_pa: int = 200,
    min_batted_balls: int = 600,
) -> PlayerProfile:
    """
    Build a PlayerProfile for batter_id using Statcast data from `seasons`.

    Parameters
    ----------
    batter_id               : Statcast (MLBAM) player id
    player_name             : display name
    seasons                 : list of seasons to pull data from
    service_time_threshold  : minimum estimated MLB seasons; below this we
                              flag is_fallback=True (actual fallback handled
                              in plate_appearance.py)
    min_pa                  : minimum total PA needed to use real profile
    min_batted_balls        : minimum batted balls needed to use real profile
    """
    from .fallback import get_fallback_profile  # avoid circular import

    df = load_batter_seasons(batter_id, seasons)
    pa_df = filter_plate_appearances(df)

    if pa_df.empty:
        reason = "No Statcast data found for this player"
        logger.warning(f"{player_name}: {reason}")
        return get_fallback_profile(batter_id, player_name, profile_reason=reason)

    # Estimate service time
    est_seasons = estimate_mlb_seasons(batter_id, seasons)
    if est_seasons < service_time_threshold:
        reason = f"Estimated MLB seasons ({est_seasons:.1f}) below threshold ({service_time_threshold:.1f})"
        logger.info(f"{player_name}: {reason} — using fallback")
        return get_fallback_profile(batter_id, player_name, profile_reason=reason)

    return _compute_profile_from_pa(pa_df, batter_id, player_name, seasons, min_pa, min_batted_balls)


def _compute_profile_from_pa(
    pa_df: pd.DataFrame,
    batter_id: int,
    player_name: str,
    seasons: List[int],
    min_pa: int = 200,
    min_batted_balls: int = 600,
) -> PlayerProfile:
    """Compute bucket weights and plate-discipline rates from a PA DataFrame."""
    from .fallback import get_fallback_profile

    n_pa = len(pa_df)
    if n_pa == 0:
        return get_fallback_profile(batter_id, player_name)

    # ---- Non-contact rates ------------------------------------------------
    events = pa_df["events"].str.lower().fillna("")

    k_mask = events.isin(["strikeout", "strikeout_double_play"])
    bb_mask = events.isin(["walk", "intent_walk"])
    hbp_mask = events.isin(["hit_by_pitch"])
    sac_mask = events.isin(["sac_fly", "sac_bunt", "sac_fly_double_play"])

    k_rate = k_mask.sum() / n_pa
    bb_rate = bb_mask.sum() / n_pa
    hbp_rate = hbp_mask.sum() / n_pa
    sac_rate = sac_mask.sum() / n_pa

    # ---- Batted-ball bucket weights ---------------------------------------
    batted = pa_df[pa_df["launch_angle"].notna() & pa_df["launch_speed"].notna()].copy()
    n_batted = len(batted)

    # Check both thresholds — need sufficient PA AND sufficient batted balls
    if n_pa < min_pa or n_batted < min_batted_balls:
        reason_parts = []
        if n_pa < min_pa:
            reason_parts.append(f"MLB PA too low ({n_pa} < {min_pa})")
        if n_batted < min_batted_balls:
            reason_parts.append(f"batted balls too low ({n_batted} < {min_batted_balls})")
        reason = " · ".join(reason_parts)
        logger.info(f"{player_name}: {reason} — using fallback")
        return get_fallback_profile(batter_id, player_name, profile_reason=reason)

    batted["_bucket"] = batted.apply(
        lambda r: assign_bucket(r["launch_angle"], r["launch_speed"]), axis=1
    )

    bucket_counts: Dict[Bucket, int] = {b: 0 for b in ALL_BUCKETS}
    for bkt in batted["_bucket"]:
        bucket_counts[bkt] += 1

    bucket_weights: Dict[Bucket, float] = {
        b: cnt / n_batted for b, cnt in bucket_counts.items()
    }

    seasons_used = sorted(pa_df["game_date"].dt.year.dropna().unique().tolist()) if "game_date" in pa_df.columns else seasons

    return PlayerProfile(
        player_id=str(batter_id),
        player_name=player_name,
        bucket_weights=bucket_weights,
        k_rate=float(k_rate),
        bb_rate=float(bb_rate),
        hbp_rate=float(hbp_rate),
        sac_rate=float(sac_rate),
        pa_count=int(n_pa),
        seasons=seasons_used,
        is_fallback=False,
    )


def summarise_profile(profile: PlayerProfile) -> pd.DataFrame:
    """Return a tidy DataFrame of non-zero bucket weights for inspection."""
    rows = []
    for bkt, w in profile.bucket_weights.items():
        if w > 0:
            rows.append({
                "la_lo": bkt.la_lo,
                "la_hi": bkt.la_hi,
                "ev_lo": bkt.ev_lo,
                "ev_hi": bkt.ev_hi,
                "la_mid": bkt.la_mid,
                "ev_mid": bkt.ev_mid,
                "weight": w,
            })
    return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)
