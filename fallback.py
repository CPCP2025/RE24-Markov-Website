"""
fallback.py
-----------
Fallback model for players with insufficient MLB history (< 200 PA or < 600 batted balls).

Strategy
--------
1. Attempt to scrape the player's MiLB batting stats via the MLB Stats API
   (statsapi.mlb.com) — same MLBAM IDs, public endpoint, no auth required.
   We request career hitting splits filtered to minor-league sport IDs 11-16.

2. Derive per-PA outcome rates from those MiLB counting stats, then apply
   scalar adjustments to project toward MLB-level expectations:
     - "Good" outcomes (1B, 2B, 3B, HR, BB, HBP) are multiplied by scalar < 1
     - STRIKEOUT rate is multiplied by scalar > 1
     - FIELD_OUT absorbs all residual probability

3. If scraping fails, fall back to the hard-coded development-level prior.

Translation scalars (MiLB → projected MLB)
------------------------------------------
Calibrated to MiLB-to-MLB translation research:
  SINGLE   0.88   (BABIP / contact regresses toward MLB average)
  DOUBLE   0.82   (extra bases translate less reliably)
  TRIPLE   0.75   (heavily park/speed dependent)
  HR       0.72   (power is hardest skill to translate)
  BB       0.90   (walk rate is relatively stable)
  HBP      0.95   (near-stable)
  K        1.12   (strikeout rate rises vs MLB-caliber pitching)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from .models import Bucket, Outcome, PlayerProfile
from .buckets import ALL_BUCKETS

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Translation scalars
# ---------------------------------------------------------------------------

GOOD_SCALARS: Dict[Outcome, float] = {
    Outcome.SINGLE:   0.88,
    Outcome.DOUBLE:   0.82,
    Outcome.TRIPLE:   0.75,
    Outcome.HOME_RUN: 0.72,
    Outcome.WALK:     0.90,
    Outcome.HBP:      0.95,
}
BAD_SCALARS: Dict[Outcome, float] = {
    Outcome.STRIKEOUT: 1.12,
}

# ---------------------------------------------------------------------------
# Hard-coded prior (last-resort fallback)
# ---------------------------------------------------------------------------

_PRIOR_K_RATE   = 0.26
_PRIOR_BB_RATE  = 0.08
_PRIOR_HBP_RATE = 0.010
_PRIOR_SAC_RATE = 0.005

_PRIOR_CONTACT_RAW: Dict[Outcome, float] = {
    Outcome.SINGLE:    0.165,
    Outcome.DOUBLE:    0.050,
    Outcome.TRIPLE:    0.006,
    Outcome.HOME_RUN:  0.030,
    Outcome.FIELD_OUT: 0.749,
}
_t = sum(_PRIOR_CONTACT_RAW.values())
_PRIOR_CONTACT: Dict[Outcome, float] = {k: v / _t for k, v in _PRIOR_CONTACT_RAW.items()}


# ---------------------------------------------------------------------------
# Neutral bucket weights
# ---------------------------------------------------------------------------

def _build_neutral_bucket_weights() -> Dict[Bucket, float]:
    eligible = [b for b in ALL_BUCKETS if -10 <= b.la_mid <= 40 and 70 <= b.ev_mid <= 105]
    if not eligible:
        eligible = list(ALL_BUCKETS)
    w = 1.0 / len(eligible)
    return {b: (w if b in eligible else 0.0) for b in ALL_BUCKETS}


_NEUTRAL_BUCKET_WEIGHTS = _build_neutral_bucket_weights()


# ---------------------------------------------------------------------------
# MLB Stats API scraping
# ---------------------------------------------------------------------------

# Primary: MLB Stats API career minor-league endpoint
# sportId (singular) is more broadly supported than sportIds
_MILB_API_URL = (
    "https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
    "?stats=career&group=hitting&sportId=11,12,13,14,15,16"
)
# Fallback: broader people endpoint that hydrates all sport stats
_MILB_API_URL2 = (
    "https://statsapi.mlb.com/api/v1/people/{player_id}"
    "?hydrate=stats(group=[hitting],type=[career],sportId=11,12,13,14,15,16)"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

MINOR_SPORT_IDS = {11, 12, 13, 14, 15, 16}


def _cache_path(player_id: int) -> Path:
    return CACHE_DIR / f"milb_stats_{player_id}.json"


def scrape_milb_stats(player_id: int, force_refresh: bool = False) -> Optional[Dict]:
    """
    Fetch career MiLB batting stats via the MLB Stats API.
    Tries two URL formats for maximum compatibility.
    Returns aggregated counting stats dict, or None on failure.
    """
    cache = _cache_path(player_id)
    if cache.exists() and not force_refresh:
        try:
            with open(cache) as f:
                data = json.load(f)
                if data.get("pa", 0) > 0:
                    return data
        except Exception:
            pass

    import urllib.request

    for url_template in [_MILB_API_URL, _MILB_API_URL2]:
        url = url_template.format(player_id=player_id)
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw = json.loads(resp.read().decode())
        except Exception as exc:
            logger.info(f"MiLB URL attempt failed for {player_id} ({url[:60]}…): {exc}")
            continue

        try:
            stats = _parse_api_response(raw)
            if not stats:
                # Try alternate structure (people hydrate endpoint)
                stats = _parse_people_response(raw)
        except Exception as exc:
            logger.info(f"MiLB parse attempt failed for {player_id}: {exc}")
            continue

        if stats and stats.get("pa", 0) >= 50:
            with open(cache, "w") as f:
                json.dump(stats, f)
            logger.info(f"MiLB data fetched for player {player_id}: {stats['pa']} PA")
            return stats

    logger.warning(
        f"MiLB data unavailable for player {player_id} — "
        f"both API endpoints returned no usable data. Using prior rates."
    )
    return None


def _parse_api_response(raw: dict) -> Dict:
    """Aggregate career MiLB hitting splits from MLB Stats API response."""
    totals = {k: 0 for k in ["pa", "ab", "h", "single", "double", "triple",
                               "hr", "bb", "hbp", "so", "sac", "sf"]}

    stat_groups = raw.get("stats", [])
    if not stat_groups:
        return totals

    splits = stat_groups[0].get("splits", [])
    for split in splits:
        sport_id = (
            split.get("sport", {}).get("id")
            or split.get("team", {}).get("sport", {}).get("id")
        )
        if sport_id is not None and int(sport_id) not in MINOR_SPORT_IDS:
            continue

        s = split.get("stat", {})
        totals["ab"]     += int(s.get("atBats", 0) or 0)
        totals["h"]      += int(s.get("hits", 0) or 0)
        totals["double"] += int(s.get("doubles", 0) or 0)
        totals["triple"] += int(s.get("triples", 0) or 0)
        totals["hr"]     += int(s.get("homeRuns", 0) or 0)
        totals["bb"]     += int(s.get("baseOnBalls", 0) or 0)
        totals["hbp"]    += int(s.get("hitByPitch", 0) or 0)
        totals["so"]     += int(s.get("strikeOuts", 0) or 0)
        totals["sac"]    += int(s.get("sacBunts", 0) or 0)
        totals["sf"]     += int(s.get("sacFlies", 0) or 0)

    totals["pa"] = (
        totals["ab"] + totals["bb"] + totals["hbp"]
        + totals["sac"] + totals["sf"]
    )
    totals["single"] = max(0, totals["h"] - totals["double"] - totals["triple"] - totals["hr"])
    return totals


def _parse_people_response(raw: dict) -> Optional[Dict]:
    """
    Parse the alternate /people endpoint hydrate response.
    Structure differs slightly from the /stats endpoint.
    """
    totals = {k: 0 for k in ["pa", "ab", "h", "single", "double", "triple",
                               "hr", "bb", "hbp", "so", "sac", "sf"]}
    people = raw.get("people", [])
    if not people:
        return None
    stats_groups = people[0].get("stats", [])
    if not stats_groups:
        return None

    for group in stats_groups:
        splits = group.get("splits", [])
        for split in splits:
            sport_id = (
                split.get("sport", {}).get("id")
                or split.get("team", {}).get("sport", {}).get("id")
            )
            if sport_id is not None and int(sport_id) not in MINOR_SPORT_IDS:
                continue
            s = split.get("stat", {})
            totals["ab"]     += int(s.get("atBats", 0) or 0)
            totals["h"]      += int(s.get("hits", 0) or 0)
            totals["double"] += int(s.get("doubles", 0) or 0)
            totals["triple"] += int(s.get("triples", 0) or 0)
            totals["hr"]     += int(s.get("homeRuns", 0) or 0)
            totals["bb"]     += int(s.get("baseOnBalls", 0) or 0)
            totals["hbp"]    += int(s.get("hitByPitch", 0) or 0)
            totals["so"]     += int(s.get("strikeOuts", 0) or 0)
            totals["sac"]    += int(s.get("sacBunts", 0) or 0)
            totals["sf"]     += int(s.get("sacFlies", 0) or 0)

    totals["pa"] = (totals["ab"] + totals["bb"] + totals["hbp"]
                    + totals["sac"] + totals["sf"])
    totals["single"] = max(0, totals["h"] - totals["double"]
                           - totals["triple"] - totals["hr"])
    return totals if totals["pa"] >= 50 else None


# ---------------------------------------------------------------------------
# Rate derivation with scalar adjustments
# ---------------------------------------------------------------------------

def _derive_rates(stats: Dict) -> Optional[Tuple]:
    """
    Convert MiLB counting stats to scaled per-PA rates.

    Returns (contact_probs, k_rate, bb_rate, hbp_rate, sac_rate) or None.
    """
    pa = stats.get("pa", 0)
    if pa < 50:
        return None

    # Raw per-PA rates
    raw: Dict[Outcome, float] = {
        Outcome.SINGLE:    max(0, stats["single"]) / pa,
        Outcome.DOUBLE:    max(0, stats["double"]) / pa,
        Outcome.TRIPLE:    max(0, stats["triple"]) / pa,
        Outcome.HOME_RUN:  max(0, stats["hr"])     / pa,
        Outcome.STRIKEOUT: max(0, stats["so"])     / pa,
        Outcome.WALK:      max(0, stats["bb"])     / pa,
        Outcome.HBP:       max(0, stats["hbp"])    / pa,
        Outcome.SAC:       max(0, stats["sac"] + stats["sf"]) / pa,
    }
    raw[Outcome.FIELD_OUT] = max(0.0, 1.0 - sum(raw.values()))

    # Apply translation scalars
    scaled: Dict[Outcome, float] = {}
    for outcome, rate in raw.items():
        if outcome in GOOD_SCALARS:
            scaled[outcome] = rate * GOOD_SCALARS[outcome]
        elif outcome in BAD_SCALARS:
            scaled[outcome] = rate * BAD_SCALARS[outcome]
        else:
            scaled[outcome] = rate

    # Field out absorbs the residual from shrinking good outcomes
    non_fo = sum(v for o, v in scaled.items() if o != Outcome.FIELD_OUT)
    scaled[Outcome.FIELD_OUT] = max(0.0, 1.0 - non_fo)

    # Normalise to exactly 1.0
    total = sum(scaled.values())
    scaled = {o: v / total for o, v in scaled.items()}

    # Extract non-contact rates
    k_rate   = scaled.pop(Outcome.STRIKEOUT, _PRIOR_K_RATE)
    bb_rate  = scaled.pop(Outcome.WALK,      _PRIOR_BB_RATE)
    hbp_rate = scaled.pop(Outcome.HBP,       _PRIOR_HBP_RATE)
    sac_rate = scaled.pop(Outcome.SAC,       _PRIOR_SAC_RATE)

    # Normalise contact distribution
    ct = sum(scaled.values())
    contact_probs = {o: v / ct for o, v in scaled.items()} if ct > 0 else _PRIOR_CONTACT.copy()

    return contact_probs, k_rate, bb_rate, hbp_rate, sac_rate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_fallback_profile(
    batter_id: int,
    player_name: str,
    force_refresh: bool = False,
    profile_reason: Optional[str] = None,
) -> PlayerProfile:
    """
    Build a fallback PlayerProfile.
    Tries MiLB scraping first; falls back to hard-coded prior on failure.

    Parameters
    ----------
    profile_reason : optional string from the caller explaining why fallback
                     was triggered (e.g. "MLB PA too low (87 < 200)").
                     Combined with the MiLB data source description into
                     profile.fallback_reason for the UI to display.
    """
    logger.info(f"Building fallback profile for {player_name} (id={batter_id})")

    milb_stats = scrape_milb_stats(batter_id, force_refresh=force_refresh)

    if milb_stats is not None:
        result = _derive_rates(milb_stats)
        if result is not None:
            contact_probs, k_rate, bb_rate, hbp_rate, sac_rate = result
            milb_pa = milb_stats.get("pa", 0)
            logger.info(
                f"{player_name}: MiLB — {milb_pa} PA | "
                f"K:{k_rate:.1%} BB:{bb_rate:.1%} HR:{contact_probs.get(Outcome.HOME_RUN,0):.2%}"
            )
            parts = []
            if profile_reason:
                parts.append(profile_reason)
            parts.append(f"Using MiLB data ({milb_pa:,} career minor-league PA) with translation scalars applied")
            reason = " · ".join(parts)
            return PlayerProfile(
                player_id=str(batter_id),
                player_name=player_name,
                bucket_weights=_NEUTRAL_BUCKET_WEIGHTS.copy(),
                k_rate=k_rate,
                bb_rate=bb_rate,
                hbp_rate=hbp_rate,
                sac_rate=sac_rate,
                pa_count=milb_pa,
                seasons=[],
                is_fallback=True,
                milb_contact_probs=contact_probs,
                fallback_reason=reason,
            )

    # MiLB API unreachable or returned no usable data
    logger.info(f"{player_name}: MiLB API unreachable — using hard-coded prior rates")
    parts = []
    if profile_reason:
        parts.append(profile_reason)
    parts.append("MiLB API unreachable — using hard-coded development-level prior rates")
    reason = " · ".join(parts)
    return PlayerProfile(
        player_id=str(batter_id),
        player_name=player_name,
        bucket_weights=_NEUTRAL_BUCKET_WEIGHTS.copy(),
        k_rate=_PRIOR_K_RATE,
        bb_rate=_PRIOR_BB_RATE,
        hbp_rate=_PRIOR_HBP_RATE,
        sac_rate=_PRIOR_SAC_RATE,
        pa_count=0,
        seasons=[],
        is_fallback=True,
        milb_contact_probs=_PRIOR_CONTACT.copy(),
        fallback_reason=reason,
    )


def get_fallback_contact_probs(profile: "PlayerProfile") -> Dict[Outcome, float]:
    """Return the contact distribution for a fallback player."""
    if hasattr(profile, "milb_contact_probs") and profile.milb_contact_probs:
        return profile.milb_contact_probs.copy()
    return _PRIOR_CONTACT.copy()


FALLBACK_RATES = {
    "k_rate":       _PRIOR_K_RATE,
    "bb_rate":      _PRIOR_BB_RATE,
    "hbp_rate":     _PRIOR_HBP_RATE,
    "sac_rate":     _PRIOR_SAC_RATE,
    "contact_rate": 1.0 - _PRIOR_K_RATE - _PRIOR_BB_RATE - _PRIOR_HBP_RATE - _PRIOR_SAC_RATE,
}
