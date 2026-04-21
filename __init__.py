"""
baseball_sim
============
A modular baseball simulation system based on Statcast batted-ball data.
"""
from .models import Bucket, Outcome, PAEvent, PlayerProfile, BaseOutState, InningResult
from .buckets import BucketModel, assign_bucket, map_event
from .ingestion import load_seasons, load_batter_seasons, search_players
from .player_profile import build_player_profile
from .fallback import get_fallback_profile
from .plate_appearance import simulate_pa
from .markov import simulate_inning, run_re24, format_re24_pivot

__all__ = [
    "Bucket", "Outcome", "PAEvent", "PlayerProfile", "BaseOutState", "InningResult",
    "BucketModel", "assign_bucket", "map_event",
    "load_seasons", "load_batter_seasons", "search_players",
    "build_player_profile", "get_fallback_profile",
    "simulate_pa", "simulate_inning", "run_re24", "format_re24_pivot",
]
