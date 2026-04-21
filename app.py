"""
app.py  —  Baseball RE24 Simulator
Run:  streamlit run app.py
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Suppress the FutureWarning from pybaseball's internal pd.to_datetime call.
# This is a pybaseball bug (not fixed as of 2.2.7); harmless but alarming to users.
warnings.filterwarnings(
    "ignore",
    message=".*errors='ignore' is deprecated.*",
    category=FutureWarning,
)

sys.path.insert(0, str(Path(__file__).parent))

from baseball_sim.buckets import BucketModel
from baseball_sim.fallback import get_fallback_profile, FALLBACK_RATES, scrape_milb_stats
from baseball_sim.ingestion import (
    load_seasons, filter_plate_appearances, search_players,
)
from baseball_sim.markov import run_re24, format_re24_pivot
from baseball_sim.models import Outcome, PlayerProfile
from baseball_sim.player_profile import build_player_profile
from baseball_sim.validation import run_validation, outcome_confusion_matrix, accuracy_by_bucket_size

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RE24 Baseball Simulator",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family:'Bebas Neue',sans-serif !important; letter-spacing:.04em; }

/* ── header ── */
.hdr { font-family:'Bebas Neue',sans-serif; font-size:3.2rem; color:#F0EBE0;
       border-bottom:4px solid #D4421A; padding-bottom:.2rem; line-height:1; }
.sub { font-family:'DM Mono',monospace; font-size:.72rem; color:#888;
       letter-spacing:.12em; text-transform:uppercase; margin-top:.3rem; margin-bottom:1.8rem; }

/* ── lineup card ── */
.lcard { background:#161625; border:1px solid #252540; border-left:4px solid #D4421A;
         border-radius:5px; padding:.85rem 1.1rem; margin-bottom:.6rem; }
.lcard-empty { background:#161625; border:1px solid #252540; border-left:4px solid #333;
               border-radius:5px; padding:.85rem 1.1rem; margin-bottom:.6rem; color:#444;
               font-style:italic; font-size:.9rem; }
.lnum { font-family:'DM Mono',monospace; font-size:.65rem; color:#888;
        text-transform:uppercase; letter-spacing:.1em; }
.lname { font-family:'Bebas Neue',sans-serif; font-size:1.35rem; color:#F0EBE0;
         line-height:1.1; }
.lmeta { font-family:'DM Mono',monospace; font-size:.68rem; color:#777; margin-top:.25rem; }

/* ── badges ── */
.badge-mlb  { background:#133a1e; color:#3dd68c; border:1px solid #2a6640;
              border-radius:3px; padding:1px 7px; font-family:'DM Mono',monospace;
              font-size:.65rem; letter-spacing:.06em; }
.badge-milb { background:#3a2800; color:#f5b731; border:1px solid #7a5200;
              border-radius:3px; padding:1px 7px; font-family:'DM Mono',monospace;
              font-size:.65rem; letter-spacing:.06em; }
.badge-prior{ background:#2a1a1a; color:#e05555; border:1px solid #5a2a2a;
              border-radius:3px; padding:1px 7px; font-family:'DM Mono',monospace;
              font-size:.65rem; letter-spacing:.06em; }

/* ── section label ── */
.sec { font-family:'Bebas Neue',sans-serif; font-size:1.45rem; color:#F0EBE0;
       letter-spacing:.05em; border-bottom:1px solid #2a2a40;
       padding-bottom:3px; margin-top:1.4rem; margin-bottom:.8rem; }

/* ── buttons ── */
.stButton>button { background:#D4421A !important; color:#fff !important;
                   font-family:'Bebas Neue',sans-serif !important; font-size:1.1rem !important;
                   letter-spacing:.07em !important; border:none !important;
                   border-radius:4px !important; padding:.45rem 1.8rem !important; }
.stButton>button:hover { background:#b03615 !important; }

/* ── sidebar ── */
[data-testid="stSidebar"] { background:#0d0d1a; }

/* ── fallback reason line ── */
.lreason { font-family:'DM Mono',monospace; font-size:.64rem; color:#c47a2a;
           margin-top:.35rem; line-height:1.4; border-top:1px solid #2a2a1a;
           padding-top:.3rem; }

/* ── tables ── */
.stDataFrame { font-family:'DM Mono',monospace; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
_SLOT_LABELS = ["Lead-off","2nd","3rd","Cleanup","5th","6th","7th","8th","9th"]

def _init():
    for k, v in [
        ("lineup",           [None]*9),
        ("bucket_model",     None),
        ("re24_result",      None),
        ("val_report",       None),
        ("search_results",   pd.DataFrame()),
        ("model_loaded",     False),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ─────────────────────────────────────────────────────────────────────────────
# Cached helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Statcast data…")
def _load_seasons(seasons: tuple) -> pd.DataFrame:
    return load_seasons(list(seasons))

@st.cache_resource(show_spinner="Building bucket model…")
def _build_model(seasons: tuple, k: float) -> BucketModel:
    df = _load_seasons(seasons)
    return BucketModel.load_or_fit(df=df, smoothing_k=k, force_refit=False)

@st.cache_data(show_spinner="Building player profile…", ttl=3600)
def _player_profile(
    batter_id: int, name: str, seasons: tuple,
    svc_thresh: float, min_pa: int, min_bb: int,
) -> PlayerProfile:
    return build_player_profile(
        batter_id=batter_id,
        player_name=name,
        seasons=list(seasons),
        service_time_threshold=svc_thresh,
        min_pa=min_pa,
        min_batted_balls=min_bb,
    )

@st.cache_data(show_spinner="Searching players…")
def _search(q: str) -> pd.DataFrame:
    return search_players(q)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — global settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚾ Global Settings")
    st.markdown("---")

    import datetime as _dt
    CUR_YEAR = _dt.date.today().year
    ALL_YEARS = list(range(2019, CUR_YEAR + 1))

    bucket_seasons = st.multiselect(
        "Bucket model seasons",
        ALL_YEARS, default=[max(2019, CUR_YEAR - 2), max(2020, CUR_YEAR - 1), CUR_YEAR],
    )
    smoothing_k = st.slider("Laplace smoothing k", 0.1, 5.0, 1.0, 0.1,
                            help="Pseudo-counts per outcome cell; prevents zero probs in sparse buckets")

    st.markdown("---")
    st.markdown("**Player lookback**")
    lookback = st.slider("Seasons of player data", 1, 5, 2)
    player_seasons = tuple(range(CUR_YEAR - lookback + 1, CUR_YEAR + 1))

    st.markdown("**Fallback thresholds**")
    svc_thresh = st.slider("Min MLB seasons (else fallback)", 0.0, 5.0, 1.0, 0.5)
    min_pa = st.number_input("Min PA for MLB profile", 50, 1000, 200, 50)
    min_bb_input = st.number_input("Min batted balls for MLB profile", 50, 3000, 200, 50)

    st.markdown("---")
    st.markdown("**Simulation**")
    n_sims = st.selectbox(
        "Simulations per state",
        [1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000],
        index=3,
        help="Higher = more accurate RE24 tables; 10,000 ≈ 10–30 s",
    )
    rand_seed = st.number_input(
        "Random seed", value=42, step=1,
        help="Fix for reproducible results; change to explore variance",
    )

    st.markdown("---")
    if st.button("Load / Rebuild Bucket Model", use_container_width=True):
        if not bucket_seasons:
            st.error("Select at least one season.")
        else:
            with st.spinner("Loading bucket model…"):
                try:
                    m = _build_model(tuple(bucket_seasons), smoothing_k)
                    st.session_state["bucket_model"] = m
                    st.session_state["model_loaded"] = True
                    st.success(f"Model ready — seasons {bucket_seasons}")
                except Exception as e:
                    st.error(f"Failed: {e}")

    if st.session_state["model_loaded"]:
        st.success("✓ Bucket model loaded")
    else:
        st.info("Click above to load model")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hdr">RE24 Baseball Simulator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Statcast Batted-Ball · 3° × 3 mph Buckets · MiLB Fallback · Markov Chain RE24</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_build, tab_sim, tab_val, tab_info = st.tabs([
    "🏟  Lineup Builder",
    "📊  Simulation",
    "🔍  Validation",
    "📖  Model Info",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LINEUP BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_build:

    # ── Player search ────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Player Search</div>', unsafe_allow_html=True)

    col_q, col_go = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "Search active MLB players",
            placeholder="e.g. Aaron Judge, Shohei, Betts…",
            label_visibility="collapsed",
        )
    with col_go:
        do_search = st.button("Search", use_container_width=True)

    if do_search and query.strip():
        # Clear previous results so stale data never lingers
        st.session_state["search_results"] = pd.DataFrame()
        results_fresh = _search(query.strip())
        if results_fresh.empty:
            st.warning(f"No players found matching '{query}'. Try a last name or partial name.")
        else:
            st.session_state["search_results"] = results_fresh

    results = st.session_state["search_results"]

    if not results.empty:
        # Build a clean selection list showing name + active years
        display = results.copy()
        first_col = display["mlb_played_first"].fillna("?").astype(str) if "mlb_played_first" in display.columns else pd.Series(["?"]*len(display))
        last_col  = display["mlb_played_last"].fillna("?").astype(str)  if "mlb_played_last"  in display.columns else pd.Series(["?"]*len(display))
        display["label"] = display["full_name"] + "  (" + first_col + "–" + last_col + ")"
        # Drop duplicate labels (same name, different IDs) — keep highest MLBAM id (most recent)
        display = display.sort_values("key_mlbam", ascending=False).drop_duplicates("label").reset_index(drop=True)

        col_pick, col_slot, col_add = st.columns([3, 2, 1])
        with col_pick:
            chosen_label = st.selectbox(
                "Select player",
                display["label"].tolist(),
                label_visibility="collapsed",
            )
        with col_slot:
            slot_choice = st.selectbox(
                "Lineup slot",
                options=list(range(1, 10)),
                format_func=lambda x: f"#{x}",
                label_visibility="collapsed",
            )
        with col_add:
            add_btn = st.button("Add ➜", use_container_width=True)

        if add_btn:
            row = display[display["label"] == chosen_label].iloc[0]
            batter_id = int(row.get("key_mlbam", 0))
            name = str(row["full_name"])

            if batter_id == 0:
                st.error("No MLBAM ID for this player — try a different name.")
            else:
                with st.spinner(f"Building profile for {name}…"):
                    try:
                        prof = _player_profile(
                            batter_id, name, player_seasons,
                            svc_thresh, int(min_pa), int(min_bb_input),
                        )
                        st.session_state["lineup"][slot_choice - 1] = prof
                        # ── Notify about fallback status immediately ──────
                        if not prof.is_fallback:
                            st.success(
                                f"✓ **{name}** added to #{slot_choice} — "
                                f"MLB profile ({prof.pa_count:,} PA across {prof.seasons})"
                            )
                        else:
                            has_milb = bool(prof.milb_contact_probs) and prof.pa_count > 0
                            reason = prof.fallback_reason or "Insufficient MLB history"
                            if has_milb:
                                st.warning(
                                    f"⚠ **{name}** added to #{slot_choice} using **MiLB data** "
                                    f"({prof.pa_count:,} minor-league PA).\n"
                                    f"*Reason: {reason}*"
                                )
                            else:
                                st.error(
                                    f"⛔ **{name}** added to #{slot_choice} using **prior rates** "
                                    f"(no MiLB data available).\n"
                                    f"*Reason: {reason}*"
                                )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Profile error: {e}")

    # ── Current lineup display ───────────────────────────────────────────────
    st.markdown('<div class="sec">Current Lineup</div>', unsafe_allow_html=True)

    lineup: List[Optional[PlayerProfile]] = st.session_state["lineup"]
    cols = st.columns(3)

    for i, prof in enumerate(lineup):
        with cols[i % 3]:
            if prof is None:
                st.markdown(
                    f'<div class="lcard-empty">#{i+1}<br>— empty —</div>',
                    unsafe_allow_html=True,
                )
            else:
                if prof.is_fallback:
                    has_milb = bool(getattr(prof, "milb_contact_probs", None)) and prof.pa_count > 0
                    badge = (
                        '<span class="badge-milb">MiLB DATA</span>'
                        if has_milb else
                        '<span class="badge-prior">PRIOR</span>'
                    )
                else:
                    badge = '<span class="badge-mlb">MLB DATA</span>'

                seasons_str = (
                    ", ".join(str(s) for s in prof.seasons[-3:])
                    if prof.seasons else "—"
                )
                reason_html = ""
                if prof.is_fallback and getattr(prof, "fallback_reason", None):
                    reason_html = (
                        f'<div class="lreason">⚠ {prof.fallback_reason}</div>'
                    )
                st.markdown(f"""
                <div class="lcard">
                  <div class="lnum">#{i+1}</div>
                  <div class="lname">{prof.player_name}</div>
                  <div class="lmeta">
                    {badge}&nbsp;
                    <span>PA:{prof.pa_count}</span>&nbsp;·&nbsp;
                    <span>K:{prof.k_rate:.1%}</span>&nbsp;·&nbsp;
                    <span>BB:{prof.bb_rate:.1%}</span>&nbsp;·&nbsp;
                    <span>Seasons:{seasons_str}</span>
                  </div>
                  {reason_html}
                </div>
                """, unsafe_allow_html=True)

    # ── Utility buttons ──────────────────────────────────────────────────────
    bu1, bu2 = st.columns(2)
    with bu1:
        if st.button("Fill empty slots with fallback players", use_container_width=True):
            for i in range(9):
                if lineup[i] is None:
                    lineup[i] = get_fallback_profile(9000 + i, f"Fallback #{i+1}")
            st.session_state["lineup"] = lineup
            st.rerun()
    with bu2:
        if st.button("Clear entire lineup", use_container_width=True):
            st.session_state["lineup"] = [None] * 9
            st.rerun()

    # ── Individual slot clear ────────────────────────────────────────────────
    with st.expander("Remove a player from a slot"):
        remove_slot = st.selectbox(
            "Slot to clear",
            range(1, 10),
            format_func=lambda x: f"#{x}  —  "
                + (lineup[x-1].player_name if lineup[x-1] else "empty"),
        )
        if st.button("Remove from slot"):
            st.session_state["lineup"][remove_slot - 1] = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown('<div class="sec">Configure & Run</div>', unsafe_allow_html=True)

    lineup = st.session_state["lineup"]
    filled = [p for p in lineup if p is not None]
    n_filled = len(filled)

    model: Optional[BucketModel] = st.session_state.get("bucket_model")
    model_ok = model is not None and model._is_fitted

    # ── Status strip ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Lineup slots filled", f"{n_filled} / 9")
    c2.metric("Bucket model", "✓ Ready" if model_ok else "✗ Not loaded")
    c3.metric("Total buckets", "1,680  (60 LA × 28 EV)" if model_ok else "—")

    st.markdown("---")

    # ── Simulation controls ──────────────────────────────────────────────────
    sim_col1, sim_col2 = st.columns([2, 1])

    with sim_col1:
        # Starting batter dropdown — shows actual player names
        batter_labels = [
            f"#{i+1}  {lineup[i].player_name if lineup[i] else '(empty)'}"
            for i in range(9)
        ]
        start_batter_raw = st.selectbox(
            "Starting batter",
            options=list(range(9)),
            format_func=lambda x: batter_labels[x],
            help="Which lineup slot leads off the simulated inning",
        )

    with sim_col2:
        st.markdown(
            f"<div style='font-family:DM Mono,monospace;font-size:.75rem;color:#888;"
            f"margin-top:1.8rem'>&#9654; {n_sims:,} sims · seed {rand_seed}</div>",
            unsafe_allow_html=True,
        )

    # ── Run button ────────────────────────────────────────────────────────────
    can_run = model_ok and n_filled == 9
    run_btn = st.button(
        f"▶  Run {n_sims:,} Simulations × 24 States",
        disabled=not can_run,
        use_container_width=True,
    )

    if not can_run:
        if not model_ok:
            st.info("↑ Load the bucket model in the sidebar first.")
        if n_filled < 9:
            st.info(f"↑ Fill all 9 lineup slots ({9 - n_filled} remaining) or use 'Fill with fallback'.")

    if run_btn:
        active = [
            lineup[i] if lineup[i] is not None
            else get_fallback_profile(9000 + i, f"Fallback #{i+1}")
            for i in range(9)
        ]
        prog = st.progress(0, text="Simulating…")
        try:
            re24 = run_re24(
                profiles=active,
                bucket_model=model,
                n_simulations=int(n_sims),
                seed=int(rand_seed),
                start_lineup_pos=int(start_batter_raw),
            )
            st.session_state["re24_result"] = re24
            prog.progress(100, text="Done!")
        except Exception as e:
            st.error(f"Simulation error: {e}")
            prog.empty()

    # ── Results ───────────────────────────────────────────────────────────────
    re24 = st.session_state.get("re24_result")
    if re24 is not None:
        st.markdown('<div class="sec">Expected Runs (RE24)</div>', unsafe_allow_html=True)
        er_pivot = format_re24_pivot(re24, "expected_runs")
        st.dataframe(
            er_pivot.style
                .background_gradient(cmap="RdYlGn", axis=None)
                .format("{:.3f}"),
            use_container_width=True,
        )

        st.markdown('<div class="sec">P(Score ≥ 1 Run)</div>', unsafe_allow_html=True)
        ps_pivot = format_re24_pivot(re24, "p_score_1plus")
        st.dataframe(
            ps_pivot.style
                .background_gradient(cmap="RdYlGn", axis=None)
                .format(lambda x: f"{x:.1%}"),
            use_container_width=True,
        )

        with st.expander("Full 24-state detail table"):
            st.dataframe(
                re24[["state_label","outs","bases","expected_runs","p_score_1plus","n_simulations"]],
                use_container_width=True,
                hide_index=True,
            )

        csv_data = re24.to_csv(index=False)
        st.download_button(
            "⬇  Download RE24 CSV",
            data=csv_data,
            file_name="re24_results.csv",
            mime="text/csv",
        )

        # ── FanGraphs MLB average reference tables ────────────────────────────
        st.markdown('<div class="sec">MLB Average Reference (FanGraphs 2021–24)</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Source: FanGraphs / Ben Clemens (May 2025). "
            "Regular-season innings 1–8, averaged across 2021–2024. "
            "Use as a benchmark against your simulated lineup."
        )

        _FG_BASES = ["---", "1B", "2B", "3B", "1B2B", "1B3B", "2B3B", "1B2B3B"]

        # Expected runs — FanGraphs 2021-24 average
        _FG_ER = {
            "---":    [0.50, 0.27, 0.10],
            "1B":     [0.90, 0.54, 0.23],
            "2B":     [1.14, 0.71, 0.33],
            "3B":     [1.37, 0.98, 0.38],
            "1B2B":   [1.51, 0.94, 0.46],
            "1B3B":   [1.82, 1.19, 0.51],
            "2B3B":   [2.04, 1.41, 0.57],
            "1B2B3B": [2.38, 1.63, 0.82],
        }
        fg_er_df = pd.DataFrame(
            _FG_ER, index=["0 Outs", "1 Out", "2 Outs"]
        ).T.loc[_FG_BASES]
        fg_er_df.index.name = "Bases"

        rfc1, rfc2 = st.columns(2)
        with rfc1:
            st.markdown("**Expected Runs — MLB Avg**")
            st.dataframe(
                fg_er_df.style
                    .background_gradient(cmap="RdYlGn", axis=None)
                    .format("{:.2f}"),
                use_container_width=True,
            )

        # Difference: simulated lineup vs MLB average
        sim_er = format_re24_pivot(re24, "expected_runs")
        # Align on same index/columns
        diff_er = sim_er.subtract(fg_er_df.rename(columns={
            "0 Outs": "0 Outs", "1 Out": "1 Out", "2 Outs": "2 Outs"
        })).round(3)
        with rfc2:
            st.markdown("**Lineup vs MLB Avg (difference)**")
            st.dataframe(
                diff_er.style
                    .background_gradient(cmap="RdYlGn", axis=None)
                    .format("{:+.3f}"),
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_val:
    st.markdown('<div class="sec">Draft Validation Mode</div>', unsafe_allow_html=True)
    st.markdown(
        "Compare the bucket model's **top-1 predicted outcome** against "
        "**observed Statcast events** on a random sample.  "
        "Use this to verify that 3°×3 mph bucket assignment and event parsing are correct."
    )

    model = st.session_state.get("bucket_model")
    if not (model and model._is_fitted):
        st.warning("Load the bucket model (sidebar) first.")
    else:
        vc1, vc2 = st.columns(2)
        with vc1:
            val_seasons = st.multiselect(
                "Validation seasons",
                ALL_YEARS,
                default=[CUR_YEAR - 1],
            )
        with vc2:
            n_val = st.slider("Sample size", 100, 3000, 500, 100)

        if st.button("Run Validation Report", use_container_width=True):
            if not val_seasons:
                st.error("Pick at least one season.")
            else:
                with st.spinner("Loading events and comparing…"):
                    try:
                        vdf = _load_seasons(tuple(val_seasons))
                        report = run_validation(vdf, model, n_sample=n_val)
                        st.session_state["val_report"] = report
                    except Exception as e:
                        st.error(f"Validation failed: {e}")

        report = st.session_state.get("val_report")
        if report is not None:
            acc = report["correct"].mean()
            vm1, vm2, vm3 = st.columns(3)
            vm1.metric("Top-1 accuracy", f"{acc:.1%}")
            vm2.metric("Sample size", f"{len(report):,}")
            vm3.metric("Distinct outcomes", report["observed"].nunique())

            st.markdown("**Per-outcome accuracy**")
            agg = (
                report.groupby("observed")["correct"]
                .agg(["mean","count"])
                .rename(columns={"mean":"accuracy","count":"n"})
                .sort_values("accuracy", ascending=False)
                .reset_index()
            )
            agg["accuracy"] = agg["accuracy"].map("{:.1%}".format)
            st.dataframe(agg, use_container_width=True, hide_index=True)

            st.markdown("**Accuracy by bucket sample size**")
            bac = accuracy_by_bucket_size(report)
            bac["accuracy"] = bac["accuracy"].map("{:.1%}".format)
            st.dataframe(bac, use_container_width=True, hide_index=True)

            st.markdown("**Confusion matrix**")
            st.dataframe(outcome_confusion_matrix(report), use_container_width=True)

            with st.expander("Raw validation rows"):
                st.dataframe(report, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.markdown("""
## Bucket Grid

| Parameter | Value |
|---|---|
| Launch angle bins | 60  (3° each, –90° to +90°) |
| Exit velocity bins | 28  (3 mph each, 40–124 mph) |
| **Total buckets** | **1,680** |
| Outcome smoothing | Laplace add-k (configurable, default k=1.0) |

## Fallback Thresholds

| Threshold | Default | Where set |
|---|---|---|
| Min MLB seasons (service time proxy) | 1.0 | Sidebar |
| Min PA for MLB profile | 200 | Sidebar |
| Min batted balls for MLB profile | 600 | Sidebar |

## MiLB Fallback Data Source

When a player falls below the thresholds the system:
1. Queries **statsapi.mlb.com** (public, no auth) for career minor-league hitting splits
2. Aggregates across sport IDs 11-16 (AAA → Rookie)
3. Applies translation scalars and re-normalises

**Translation scalars applied:**

| Outcome | Scalar | Direction |
|---|---|---|
| Single | 0.88 | ↓ |
| Double | 0.82 | ↓ |
| Triple | 0.75 | ↓ |
| Home Run | 0.72 | ↓ (hardest to translate) |
| Walk | 0.90 | ↓ |
| HBP | 0.95 | ≈ stable |
| Strikeout | 1.12 | ↑ (harder MLB pitching) |
| Field Out | residual | absorbs all removed probability |

If the API is unreachable, the prior below is used instead.

## Hard-coded Prior (last resort)

| Rate | Value |
|---|---|
| K% | 26% |
| BB% | 8% |
| HBP% | 1% |
| SAC% | 0.5% |
| Contact% | 64.5% |

## Runner Advancement

Standard simplified Markov assumptions:
- **Single**: 3B/2B score; 1B → 3B; batter to 1B
- **Double**: all score; batter to 2B
- **Triple**: all score; batter to 3B
- **HR**: batter + all score; bases clear
- **Walk/HBP**: forced advancement only
- **K/FO**: out recorded; no advancement
- **SAC**: lead runner advances; out recorded
""")

    st.markdown("**Prior fallback rates**")
    st.dataframe(
        pd.DataFrame.from_dict(FALLBACK_RATES, orient="index", columns=["Rate"])
        .style.format("{:.1%}"),
        use_container_width=False,
    )
