"""
markov.py
---------
Recursive Markov-chain inning simulator.

Each call to simulate_inning():
  1. Starts from a given BaseOutState.
  2. Loops until 3 outs are recorded.
  3. Uses the correct batter from the fixed lineup each PA.
  4. Updates base-runner state using simplified MLB runner-advancement rules.
  5. Returns (runs_scored, at_least_one_run_scored, final_lineup_pos).

RE24 computation (run_re24()):
  For each of the 24 base-out states, run N simulated innings starting from
  that state (with outs already in that state, continuing the *same* inning).
  Aggregate:
    * expected_runs  = mean(runs_scored across simulations)
    * p_score_1plus  = mean(ran at least 1 run)

Runner advancement rules (simplified)
--------------------------------------
These are the standard simplified Markov-chain advancement assumptions:
  SINGLE : all runners advance 2 bases (+ batter to 1B)
  DOUBLE : all runners score; batter to 2B
  TRIPLE : all runners score; batter to 3B
  HR     : batter + all runners score; bases empty
  FIELD_OUT : no runners advance (conservative — ignores sac fly scoring)
  STRIKEOUT : no runners advance
  WALK / HBP : forced advancement only
  SAC    : 0 outs consumed, lead runner advances 1 base (simplified)

These rules are well-known simplifications; they undercount run production
slightly because they don't model tag-ups or first-to-third on singles.
A more realistic model would require a separate base-running lookup table.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .buckets import BucketModel
from .models import BaseOutState, InningResult, Outcome, PlayerProfile
from .plate_appearance import simulate_pa

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base-runner advancement
# ---------------------------------------------------------------------------

def advance_runners(
    state: BaseOutState,
    outcome: Outcome,
) -> Tuple[BaseOutState, int]:
    """
    Apply an outcome to a BaseOutState.

    Returns
    -------
    (new_state, runs_scored_on_this_play)
    """
    on1, on2, on3 = state.bases
    outs = state.outs
    runs = 0

    if outcome == Outcome.HOME_RUN:
        runs = 1 + int(on1) + int(on2) + int(on3)
        return BaseOutState(bases=(False, False, False), outs=outs), runs

    if outcome == Outcome.TRIPLE:
        runs = int(on1) + int(on2) + int(on3)
        return BaseOutState(bases=(False, False, True), outs=outs), runs

    if outcome == Outcome.DOUBLE:
        runs = int(on2) + int(on3)
        # runner on 1B scores on double (aggressive assumption)
        if on1:
            runs += 1
        return BaseOutState(bases=(False, True, False), outs=outs), runs

    if outcome == Outcome.SINGLE:
        # Simplified: runner on 3B scores; 2B scores; 1B goes to 3B
        runs = int(on3) + int(on2)
        new1 = True          # batter
        new2 = False
        new3 = on1           # runner who was on 1B advances to 3B
        return BaseOutState(bases=(new1, new2, new3), outs=outs), runs

    if outcome in (Outcome.WALK, Outcome.HBP):
        # Forced advancement
        if on1 and on2 and on3:
            runs = 1
            return BaseOutState(bases=(True, True, True), outs=outs), runs
        elif on1 and on2:
            return BaseOutState(bases=(True, True, True), outs=outs), 0
        elif on1:
            return BaseOutState(bases=(True, True, on3), outs=outs), 0
        else:
            return BaseOutState(bases=(True, on2, on3), outs=outs), 0

    if outcome == Outcome.SAC:
        # Simplified sac: lead runner advances; out is recorded
        new_outs = outs + 1
        if on3:
            runs = 1
            return BaseOutState(bases=(on1, on2, False), outs=new_outs), runs
        elif on2:
            return BaseOutState(bases=(on1, False, True), outs=new_outs), 0
        elif on1:
            return BaseOutState(bases=(False, True, on3), outs=new_outs), 0
        else:
            # No runner to advance; still an out
            return BaseOutState(bases=(on1, on2, on3), outs=new_outs), 0

    if outcome in (Outcome.FIELD_OUT, Outcome.STRIKEOUT):
        new_outs = outs + 1
        return BaseOutState(bases=(on1, on2, on3), outs=new_outs), 0

    # Unknown outcome: treat as field out
    return BaseOutState(bases=(on1, on2, on3), outs=outs + 1), 0


# ---------------------------------------------------------------------------
# Single inning simulation
# ---------------------------------------------------------------------------

def simulate_inning(
    profiles: List[PlayerProfile],
    bucket_model: BucketModel,
    rng: np.random.Generator,
    start_state: Optional[BaseOutState] = None,
    start_lineup_pos: int = 0,
) -> InningResult:
    """
    Simulate one half-inning starting from start_state.

    Parameters
    ----------
    profiles         : list of 9 PlayerProfiles (lineup order)
    bucket_model     : fitted BucketModel
    rng              : random generator
    start_state      : initial BaseOutState (default: bases empty, 0 outs)
    start_lineup_pos : batter index (0–8) at start of inning

    Returns
    -------
    InningResult with runs scored and final lineup position
    """
    if start_state is None:
        start_state = BaseOutState.empty()

    state = start_state
    lineup_pos = start_lineup_pos % len(profiles)
    runs = 0

    while state.outs < 3:
        profile = profiles[lineup_pos % len(profiles)]
        pa = simulate_pa(profile, bucket_model, rng)
        state, runs_scored = advance_runners(state, pa.outcome)
        runs += runs_scored
        lineup_pos += 1

    return InningResult(
        runs=runs,
        scored=(runs > 0),
        starting_state=start_state,
        ending_lineup_pos=lineup_pos % len(profiles),
    )


# ---------------------------------------------------------------------------
# RE24 computation
# ---------------------------------------------------------------------------

def run_re24(
    profiles: List[PlayerProfile],
    bucket_model: BucketModel,
    n_simulations: int = 10_000,
    seed: Optional[int] = 42,
    start_lineup_pos: int = 0,
) -> pd.DataFrame:
    """
    Compute RE24-style tables for all 24 base-out states.

    For each state, we simulate N innings starting with that state's out count
    already in effect (i.e. we are continuing an inning from that state).

    Returns
    -------
    DataFrame with columns:
        state_label, outs, bases, state_index,
        expected_runs, p_score_1plus, n_simulations
    Sorted by state_index.
    """
    rng = np.random.default_rng(seed)

    # All 24 states
    all_states: List[BaseOutState] = []
    for outs in range(3):
        for b0 in [False, True]:
            for b1 in [False, True]:
                for b2 in [False, True]:
                    all_states.append(BaseOutState(bases=(b0, b1, b2), outs=outs))

    results = []
    for state in all_states:
        run_totals = []
        scored_flags = []

        for _ in range(n_simulations):
            result = simulate_inning(
                profiles=profiles,
                bucket_model=bucket_model,
                rng=rng,
                start_state=state,
                start_lineup_pos=start_lineup_pos,
            )
            run_totals.append(result.runs)
            scored_flags.append(int(result.scored))

        exp_runs = float(np.mean(run_totals))
        p_score = float(np.mean(scored_flags))

        bases_label = (
            "".join(lbl for lbl, on in zip(["1B", "2B", "3B"], state.bases) if on)
            or "---"
        )
        results.append({
            "state_index": state.state_index,
            "outs": state.outs,
            "bases": bases_label,
            "state_label": f"{bases_label}, {state.outs} out",
            "expected_runs": round(exp_runs, 3),
            "p_score_1plus": round(p_score, 3),
            "n_simulations": n_simulations,
        })

    df = pd.DataFrame(results).sort_values("state_index").reset_index(drop=True)
    return df


def format_re24_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Reshape RE24 results into an 8 × 3 pivot table (bases × outs).
    Rows = base configuration, Columns = 0 / 1 / 2 outs.
    """
    pivot = df.pivot(index="bases", columns="outs", values=value_col)
    row_order = ["---", "1B", "2B", "3B", "1B2B", "1B3B", "2B3B", "1B2B3B"]
    row_order = [r for r in row_order if r in pivot.index]
    pivot = pivot.loc[row_order]
    pivot.columns = [f"{o} Out{'s' if o != 1 else ''}" for o in sorted(pivot.columns)]
    pivot.index.name = "Bases"
    return pivot
