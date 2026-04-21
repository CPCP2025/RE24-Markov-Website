"""
ingestion.py
------------
Handles all data retrieval from pybaseball (Statcast) with local disk caching.
Results are stored as Parquet files under data/ so repeated runs are fast.

Assumptions
-----------
* We use statcast() or statcast_batter() from pybaseball.
* All date logic is calendar-year-based; "season N" = Apr 1 – Oct 31 of year N.
* Only events with a valid launch_angle AND launch_speed (exit velo) are
  considered "batted-ball" events; all other events (K, BB, HBP, etc.) are
  retained with null LA/EV.
"""
from __future__ import annotations

import logging
import os
import warnings
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

# pybaseball internally calls pd.to_datetime with errors='ignore', which is
# deprecated in pandas >= 2.0 and emits a noisy FutureWarning on every scrape.
# Suppressed here at the source; remove once pybaseball fixes this upstream.
warnings.filterwarnings(
    "ignore",
    message=".*errors='ignore' is deprecated.*",
    category=FutureWarning,
)

logger = logging.getLogger(__name__)

# Where we store cached files
CACHE_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Columns we actually need (keeps file sizes small)
KEEP_COLS = [
    "game_date",
    "batter",
    "pitcher",
    "events",          # outcome label (single, double, home_run, strikeout, …)
    "description",
    "bb_type",         # batted ball type (line_drive, fly_ball, …)
    "launch_angle",
    "launch_speed",    # exit velocity
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "player_name",
    "stand",
    "p_throws",
    "inning",
    "inning_topbot",
    "home_team",
    "away_team",
    "at_bat_number",
    "pitch_number",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "post_home_score",
    "post_away_score",
    "delta_run_exp",
]


def _season_cache_path(season: int) -> Path:
    return CACHE_DIR / f"statcast_{season}.parquet"


def _batter_cache_path(batter_id: int, season: int) -> Path:
    return CACHE_DIR / f"batter_{batter_id}_{season}.parquet"


def load_season(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load a full season of Statcast data, using a local cache if available.

    Parameters
    ----------
    season       : MLB season year (e.g. 2023)
    force_refresh: if True, bypass cache and re-scrape

    Returns
    -------
    DataFrame with one row per pitch/event, trimmed to KEEP_COLS.
    """
    cache = _season_cache_path(season)
    if cache.exists() and not force_refresh:
        logger.info(f"Loading season {season} from cache: {cache}")
        return pd.read_parquet(cache)

    logger.info(f"Scraping Statcast for season {season}...")
    try:
        import pybaseball as pyb
        pyb.cache.enable()  # enable pybaseball's own caching
        df = pyb.statcast(
            start_dt=f"{season}-04-01",
            end_dt=f"{season}-10-31",
        )
    except Exception as exc:
        logger.error(f"Failed to scrape season {season}: {exc}")
        raise

    df = _trim_and_clean(df)
    df.to_parquet(cache, index=False)
    logger.info(f"Saved season {season} to {cache} ({len(df):,} rows)")
    return df


def load_seasons(seasons: List[int], force_refresh: bool = False) -> pd.DataFrame:
    """Concatenate multiple seasons."""
    frames = [load_season(s, force_refresh=force_refresh) for s in seasons]
    return pd.concat(frames, ignore_index=True)


def load_batter_seasons(
    batter_id: int,
    seasons: List[int],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load Statcast data for a specific batter across given seasons.
    Uses per-batter per-season cache files for efficiency.
    """
    frames = []
    for season in seasons:
        cache = _batter_cache_path(batter_id, season)
        if cache.exists() and not force_refresh:
            logger.info(f"Loading batter {batter_id} season {season} from cache")
            frames.append(pd.read_parquet(cache))
            continue

        logger.info(f"Scraping batter {batter_id} season {season}...")
        try:
            import pybaseball as pyb
            pyb.cache.enable()
            df = pyb.statcast_batter(
                start_dt=f"{season}-04-01",
                end_dt=f"{season}-10-31",
                player_id=batter_id,
            )
        except Exception as exc:
            logger.warning(f"Could not scrape batter {batter_id} season {season}: {exc}")
            frames.append(pd.DataFrame())
            continue

        df = _trim_and_clean(df)
        df.to_parquet(cache, index=False)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _get_chadwick_table() -> pd.DataFrame:
    """
    Load the Chadwick Bureau lookup table (pybaseball's internal cache).
    Falls back to downloading via pybaseball if not cached locally.
    Returns a DataFrame with lowercased name columns.
    """
    local_cache = CACHE_DIR / "chadwick_register.parquet"
    if local_cache.exists():
        return pd.read_parquet(local_cache)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from pybaseball.playerid_lookup import get_lookup_table
        df = get_lookup_table()  # already lowercased; downloads if not cached

    df.to_parquet(local_cache, index=False)
    return df


def search_players(name_query: str) -> pd.DataFrame:
    """
    Search for players by name against the Chadwick Bureau register.

    Strategy (in order, stops at first non-empty result):
      1. Exact full-name match  ("tj friedl" → first=tj last=friedl)
      2. Substring match on last name + optional first-name prefix filter
         ("betts" → all rows where name_last contains "betts")
      3. Substring match on first name alone
         ("tj" → rows where name_first starts with "tj")
      4. Difflib fuzzy match on "firstname lastname" combined string
         (handles typos / close spellings; top-10 by similarity score)

    Returns a DataFrame with columns:
        full_name, key_mlbam, mlb_played_first, mlb_played_last
    """
    KEEP = ["name_first", "name_last", "key_mlbam",
            "key_fangraphs", "mlb_played_first", "mlb_played_last"]

    try:
        table = _get_chadwick_table()
    except Exception as exc:
        logger.error(f"Could not load Chadwick register: {exc}")
        return pd.DataFrame()

    table = table.copy()
    # Ensure columns exist and are lowercase strings
    table["name_first"] = table["name_first"].fillna("").str.lower().str.strip()
    table["name_last"] = table["name_last"].fillna("").str.lower().str.strip()
    table["_full"] = table["name_first"] + " " + table["name_last"]

    query = name_query.strip().lower()
    tokens = query.split()
    if not tokens:
        return pd.DataFrame()

    result = pd.DataFrame()

    # ---- Strategy 1: exact full-name or last-name match ------------------
    if len(tokens) >= 2:
        # Try "firstname lastname" and also "lastname firstname"
        first_a, last_a = " ".join(tokens[:-1]), tokens[-1]
        first_b, last_b = " ".join(tokens[1:]), tokens[0]
        mask = (
            ((table["name_last"] == last_a) & (table["name_first"] == first_a)) |
            ((table["name_last"] == last_b) & (table["name_first"] == first_b))
        )
        result = table[mask]

    if result.empty:
        # Exact last-name match (handles single token like "betts")
        last_token = tokens[-1]
        mask = table["name_last"] == last_token
        if len(tokens) >= 2:
            first_prefix = " ".join(tokens[:-1])
            mask = mask & table["name_first"].str.startswith(first_prefix)
        result = table[mask]

    # ---- Strategy 2: substring on last name ------------------------------
    if result.empty:
        last_token = tokens[-1]
        mask = table["name_last"].str.contains(last_token, regex=False)
        if len(tokens) >= 2:
            first_prefix = " ".join(tokens[:-1])
            mask = mask & table["name_first"].str.startswith(first_prefix)
        result = table[mask]

    # ---- Strategy 3: substring on first name (single token only) ---------
    if result.empty and len(tokens) == 1:
        result = table[table["name_first"].str.contains(tokens[0], regex=False)]

    # ---- Strategy 4: difflib fuzzy on combined "first last" string -------
    if result.empty:
        from difflib import get_close_matches
        combined = (table["_full"]).tolist()
        matches = get_close_matches(query, combined, n=10, cutoff=0.5)
        if not matches:
            # Loosen cutoff for very short names / initials
            matches = get_close_matches(query, combined, n=10, cutoff=0.3)
        if matches:
            result = table[table["_full"].isin(matches)]

    if result.empty:
        logger.info(f"No results for {name_query!r}")
        return pd.DataFrame()

    result = result[[c for c in KEEP if c in result.columns]].copy()
    result["full_name"] = (
        result["name_first"].str.strip().str.title()
        + " "
        + result["name_last"].str.strip().str.title()
    )

    # Parse year columns to numeric first (keeps them sortable as Int64)
    for _yr_col in ("mlb_played_first", "mlb_played_last"):
        if _yr_col in result.columns:
            result[_yr_col] = pd.to_numeric(result[_yr_col], errors="coerce")

    # Filter to players who have actually appeared in MLB (numeric year > 0)
    if "mlb_played_first" in result.columns:
        result = result[result["mlb_played_first"].notna() & (result["mlb_played_first"] > 0)]

    # Drop rows with no MLBAM id (can't scrape Statcast without it)
    result = result[result["key_mlbam"].notna() & (result["key_mlbam"] != 0)]
    result["key_mlbam"] = result["key_mlbam"].astype(int)

    # Sort by most recent MLB appearance (numeric column — safe to sort now)
    result = result.sort_values("mlb_played_last", ascending=False).reset_index(drop=True)

    # Now convert years to display strings (after sorting, so no mixed-type sort crash)
    for _yr_col in ("mlb_played_first", "mlb_played_last"):
        if _yr_col in result.columns:
            result[_yr_col] = result[_yr_col].apply(
                lambda v: str(int(v)) if pd.notna(v) and v > 0 else "?"
            )

    return result


def get_player_pa_count(batter_id: int, seasons: List[int]) -> int:
    """Count total plate appearances for a batter across seasons."""
    try:
        df = load_batter_seasons(batter_id, seasons)
        if df.empty:
            return 0
        # Each row is a pitch; filter to final pitches (events not null)
        return int(df["events"].notna().sum())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trim_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Subset to relevant columns, parse dates, cast types."""
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    if "launch_speed" in df.columns:
        df["launch_speed"] = pd.to_numeric(df["launch_speed"], errors="coerce")

    if "launch_angle" in df.columns:
        df["launch_angle"] = pd.to_numeric(df["launch_angle"], errors="coerce")

    return df


def filter_plate_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows that represent the end of a plate appearance
    (i.e., rows where 'events' is not null/empty).
    """
    if df.empty:
        return df
    return df[df["events"].notna() & (df["events"] != "")].copy()


def filter_batted_balls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows that are batted-ball events (have valid LA and EV).
    """
    if df.empty:
        return df
    return df[
        df["launch_angle"].notna() & df["launch_speed"].notna()
    ].copy()
