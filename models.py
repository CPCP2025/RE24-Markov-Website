"""
models.py
---------
Shared dataclasses and typed models for the baseball simulation system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Outcome labels
# ---------------------------------------------------------------------------

class Outcome(str, Enum):
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    HOME_RUN = "home_run"
    FIELD_OUT = "field_out"
    STRIKEOUT = "strikeout"
    WALK = "walk"
    HBP = "hbp"
    SAC = "sac"

    @property
    def is_hit(self) -> bool:
        return self in (Outcome.SINGLE, Outcome.DOUBLE, Outcome.TRIPLE, Outcome.HOME_RUN)

    @property
    def bases(self) -> int:
        """Number of bases for a hit outcome."""
        return {
            Outcome.SINGLE: 1,
            Outcome.DOUBLE: 2,
            Outcome.TRIPLE: 3,
            Outcome.HOME_RUN: 4,
        }.get(self, 0)


# ---------------------------------------------------------------------------
# Bucket definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bucket:
    """Represents one (launch_angle, exit_velocity) cell."""
    la_lo: float
    la_hi: float
    ev_lo: float
    ev_hi: float

    @property
    def la_mid(self) -> float:
        return (self.la_lo + self.la_hi) / 2

    @property
    def ev_mid(self) -> float:
        return (self.ev_lo + self.ev_hi) / 2

    def contains(self, la: float, ev: float) -> bool:
        return self.la_lo <= la < self.la_hi and self.ev_lo <= ev < self.ev_hi

    def __repr__(self) -> str:
        return f"Bucket(LA={self.la_lo:.0f}–{self.la_hi:.0f}, EV={self.ev_lo:.0f}–{self.ev_hi:.0f})"


# ---------------------------------------------------------------------------
# Plate-appearance event
# ---------------------------------------------------------------------------

@dataclass
class PAEvent:
    """A single plate-appearance event, either observed or simulated."""
    player_id: str
    player_name: str
    outcome: Outcome
    launch_angle: Optional[float] = None
    exit_velocity: Optional[float] = None
    bucket: Optional[Bucket] = None
    # For validation: the original event_type / bb_type from Statcast
    raw_event: Optional[str] = None
    raw_bb_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Player profile
# ---------------------------------------------------------------------------

@dataclass
class PlayerProfile:
    """
    All information needed to simulate plate appearances for one batter.

    bucket_weights      : dict mapping Bucket -> fraction of batted balls (sums ~1)
    k_rate              : strikeout rate (per PA)
    bb_rate             : walk rate (per PA)
    hbp_rate            : hit-by-pitch rate (per PA)
    sac_rate            : sacrifice rate (per PA) — usually very small
    pa_count            : total PAs in sample used to build this profile
    seasons             : list of seasons included
    is_fallback         : True if we used the youth/MiLB fallback model
    milb_contact_probs  : for fallback players, the scaled MiLB contact
                          distribution {Outcome -> prob}; None for MLB players
    fallback_reason     : human-readable explanation of why fallback was used
    """
    player_id: str
    player_name: str
    bucket_weights: Dict[Bucket, float]
    k_rate: float
    bb_rate: float
    hbp_rate: float
    sac_rate: float
    pa_count: int
    seasons: List[int]
    is_fallback: bool = False
    milb_contact_probs: Optional[Dict] = None
    fallback_reason: Optional[str] = None

    # Pre-computed arrays for fast simulation — built once in __post_init__,
    # never rebuilt per-PA.  Not part of the public API.
    _bucket_list: List = field(default_factory=list, repr=False, compare=False)
    _bucket_weights_arr: Optional[object] = field(default=None, repr=False, compare=False)

    # Pre-computed PA branch probabilities [K, BB, HBP, SAC, contact]
    _pa_probs: Optional[object] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Pre-compute all hot-path arrays so simulate_pa is pure array ops."""
        import numpy as np
        # Bucket arrays
        if self.bucket_weights:
            self._bucket_list = list(self.bucket_weights.keys())
            w = np.array(list(self.bucket_weights.values()), dtype=np.float64)
            total = w.sum()
            self._bucket_weights_arr = w / total if total > 0 else np.ones(len(w)) / len(w)
        # PA branch probabilities [K, BB, HBP, SAC, contact]
        rates = np.array([self.k_rate, self.bb_rate, self.hbp_rate, self.sac_rate], dtype=np.float64)
        contact = max(0.0, 1.0 - rates.sum())
        pa_arr = np.append(rates, contact)
        total = pa_arr.sum()
        self._pa_probs = pa_arr / total if total > 0 else np.array([0, 0, 0, 0, 1.0])


# ---------------------------------------------------------------------------
# Base-out state
# ---------------------------------------------------------------------------

@dataclass
class BaseOutState:
    """
    24-state representation: 8 base configurations × 3 out counts.
    Bases are represented as a tuple of booleans (1B, 2B, 3B).
    """
    bases: Tuple[bool, bool, bool]  # (on_first, on_second, on_third)
    outs: int  # 0, 1, or 2

    def __post_init__(self):
        assert 0 <= self.outs <= 3, "Outs must be 0, 1, 2, or 3 (3 = inning over)"
        assert len(self.bases) == 3

    @classmethod
    def empty(cls) -> "BaseOutState":
        return cls(bases=(False, False, False), outs=0)

    @property
    def state_index(self) -> int:
        """Unique integer 0–23 for this state."""
        base_idx = (
            self.bases[0] * 4 + self.bases[1] * 2 + self.bases[2]
        )
        return self.outs * 8 + base_idx

    @property
    def runners_on(self) -> int:
        return sum(self.bases)

    def __repr__(self) -> str:
        b = "".join(
            label for label, on in zip(["1B", "2B", "3B"], self.bases) if on
        ) or "---"
        return f"State({b}, {self.outs} out)"


# ---------------------------------------------------------------------------
# Inning result
# ---------------------------------------------------------------------------

@dataclass
class InningResult:
    runs: int
    scored: bool  # at least one run scored
    starting_state: BaseOutState
    ending_lineup_pos: int  # next batter index (0-8)
