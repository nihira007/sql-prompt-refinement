"""
app.py -- Streamlit dashboard for "Iterative Prompt Refinement for Automated
SQL Query Planning via TextGrad".

Run with:   streamlit run app.py

Tabs:
  1. Live refinement  : ask a question, watch  SQL -> execution -> query plan -> TextGrad
                        feedback -> refined prompt  round by round (built-in or uploaded DB).
  2. Benchmark        : static zero-shot / few-shot vs a TextGrad-optimised prompt on 3
                        complexity tiers, with confidence intervals and significance tests.
  3. How it works     : the idea, the metrics and the design choices in one page.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import database as db
from benchmark import LABEL_TO_TIER, SIZES, TIER_DESCRIPTIONS, TIER_LABELS, items_for
from llm_client import LLMError, LLMStatus, get_llm
from pipeline import (
    FEW_SHOT_PROMPT,
    ZERO_SHOT_PROMPT,
    AutoRefiningPipeline,
    RefinementRound,
    TierBenchmark,
    benchmark_frame,
    ci_frame,
    estimate_calls,
    paired_frame,
    per_question_frame,
    prompt_diff,
    seed_summary,
    taxonomy_frame,
)
from report import build_json_report, build_markdown_report
from stats import format_ci

try:  # Altair ships with Streamlit; the chart code below degrades gracefully without it
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

st.set_page_config(page_title="TextGrad SQL Prompt Refinement", page_icon="🧬", layout="wide")


# --------------------------------------------------------------------------- #
# Theme-aware visual styling
# Uses Streamlit's theme variables so inputs and surfaces adapt to light/dark.
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
    /* Base layout */
    .stApp {
        background: var(--background-color);
        color: var(--text-color);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: var(--secondary-background-color);
        border-right: 1px solid var(--border-color, rgba(128, 128, 128, 0.22));
    }

    [data-testid="stSidebar"] > div:first-child {
        background: var(--secondary-background-color);
    }

    /* Typography and spacing */
    h1, h2, h3, h4, h5, h6,
    p, label, [data-testid="stCaptionContainer"] {
        color: var(--text-color);
    }

    h1 {
        letter-spacing: -0.035em;
        font-weight: 700;
    }

    h2, h3 {
        letter-spacing: -0.02em;
        font-weight: 650;
    }

    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: var(--text-color);
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--primary-color);
    }

    /* Text inputs, text areas, number/date inputs and select boxes */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    .stDateInput input,
    .stTimeInput input,
    div[data-baseweb="select"] > div {
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.35)) !important;
        border-radius: 9px !important;
        box-shadow: none !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder,
    .stNumberInput input::placeholder {
        color: var(--text-color) !important;
        opacity: 0.58;
    }

    .stTextArea textarea {
        line-height: 1.5;
    }

    /* Select menus and popovers */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border-color: var(--border-color, rgba(128, 128, 128, 0.35)) !important;
    }

    li[role="option"],
    div[data-baseweb="option"] {
        color: var(--text-color) !important;
    }

    li[role="option"]:hover,
    div[data-baseweb="option"]:hover {
        background-color: var(--background-color) !important;
    }

    /* Radio buttons, checkboxes and sliders */
    .stRadio label,
    .stCheckbox label,
    .stSlider label {
        color: var(--text-color) !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
        transition: background-color 120ms ease, border-color 120ms ease;
    }

    .stButton > button[kind="primary"] {
        background-color: var(--primary-color, #285943) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--primary-color, #285943) !important;
        opacity: 1 !important;
    }

    .stButton > button[kind="primary"] p {
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover {
        filter: brightness(0.92);
        color: #FFFFFF !important;
    }

    .stButton > button:focus,
    .stDownloadButton > button:focus {
        box-shadow: 0 0 0 2px var(--primary-color) !important;
    }

    /* Expanders, alerts and metric containers */
    [data-testid="stExpander"] {
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.28));
        border-radius: 10px;
        background: var(--secondary-background-color);
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.24));
        border-radius: 10px;
        padding: 14px 16px;
    }

    /* Tables and code blocks */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border-radius: 9px;
        overflow: hidden;
    }

    [data-testid="stCode"],
    pre {
        border-radius: 9px;
    }

    /* Progress bar */
    [data-testid="stProgressBar"] > div > div {
        background-color: var(--primary-color);
    }

    /* Keep the layout comfortable on smaller screens */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Small compatibility helpers 

_ver = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
STRETCH: dict = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}
CHART_STRETCH: dict = {"width": "stretch"} if _ver >= (1, 50) else {}

CUSTOM_OPTION = "✎ Write my own question…"
DEMO_CAVEAT = (
    "**Demo mode.** No model is configured, so an offline *simulator* plays the LLM. It is scripted to make "
    "typical text-to-SQL mistakes that a prompt guideline can fix, so the numbers below prove the pipeline "
    "(error and plan capture → textual gradient → prompt rewrite → validation) works, **not** how a real model "
    "performs. Configure a model (see the README) for real measurements."
)


def show_table(df: pd.DataFrame) -> None:
    st.dataframe(df, hide_index=True, **STRETCH)


def pct(x: float) -> str:
    return f"{x:.0%}"


def save_upload(upload) -> str:
    """Store an uploaded SQLite file under a content-hash name in the temp folder and return its path."""
    data = upload.getvalue()
    path = Path(tempfile.gettempdir()) / f"textgrad_sql_upload_{hashlib.sha256(data).hexdigest()[:16]}.db"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


# Cached resources


@st.cache_resource(show_spinner=False)
def build_pipeline(prefer: str, db_path: str) -> AutoRefiningPipeline:
    # Key, model and base URL come from API_KEY / MODEL / BASE_URL (or a .env file).
    return AutoRefiningPipeline(get_llm(prefer=prefer), db_path=db_path or None)


# Sidebar

st.sidebar.title("⏣ Settings")

st.sidebar.subheader("Data")
source = st.sidebar.radio("Data source", ["Built-in e-commerce database", "Upload my own SQLite file"])
custom_path = ""
if source.startswith("Upload"):
    upload = st.sidebar.file_uploader("SQLite file", type=["db", "sqlite", "sqlite3"])
    if upload is None:
        st.title("Iterative Prompt Refinement for SQL Query Planning via TextGrad")
        st.info("Upload a SQLite file in the sidebar, or switch back to the built-in e-commerce database.")
        st.stop()
    custom_path = save_upload(upload)
    ok, message = db.validate_sqlite_file(custom_path)
    if not ok:
        st.sidebar.error(message)
        st.stop()
    st.sidebar.success(message)
is_builtin = not custom_path

tier = None
tier_label = ""
if is_builtin:
    tier_label = st.sidebar.radio(
        "Query complexity tier", list(TIER_LABELS.values()),
        help="Single-Table: one table. Multi-Join: 2-4 joined tables. Advanced: subqueries, CTEs, window functions.",
    )
    tier = LABEL_TO_TIER[tier_label]

st.sidebar.subheader("Language model")
mode_choice = st.sidebar.selectbox("Mode", ["Auto (use configured model)", "Demo (offline simulator)"])
pipeline = build_pipeline("demo" if mode_choice.startswith("Demo") else "auto", custom_path)
llm_status = LLMStatus.of(pipeline.llm)
(st.sidebar.success if llm_status.mode == "live" else st.sidebar.warning)(llm_status.notice)

plan_aware = st.sidebar.checkbox(
    "Penalise inefficient query plans", value=True,
    help="A correct query that needs several times the work of the reference (measured in SQLite VM steps, "
         "e.g. a correlated subquery) counts as a failure and is fed back to TextGrad.",
)
pipeline.plan_aware = bool(plan_aware)
if hasattr(pipeline.llm, "clear_cache"):
    st.sidebar.caption(f"Response cache: {pipeline.llm.entries()} entries (runs resume where they stopped)")
    if st.sidebar.button("Clear response cache"):
        st.sidebar.success(f"Removed {pipeline.llm.clear_cache()} cached responses.")

st.sidebar.subheader("TextGrad loop")
max_rounds = st.sidebar.slider("Max rounds per question (live view)", 1, 6, 4)
epochs = st.sidebar.slider("Training epochs (benchmark)", 1, 4, 2)
batch_size = st.sidebar.slider("Mini-batch size (benchmark)", 1, 6, 3)
size_choice = st.sidebar.selectbox(
    "Benchmark size", list(SIZES),
    format_func=lambda k: f"{k} ({SIZES[k][0]} train / {SIZES[k][1]} test per tier)",
    help="Quick keeps the number of model calls low (good for local models).",
)
n_seeds = st.sidebar.slider("Optimisation runs (seeds)", 1, 5, 1)

st.sidebar.subheader("Database")
st.sidebar.dataframe(
    pd.DataFrame(db.get_table_overview(pipeline.db_path).items(), columns=["table", "rows"]), hide_index=True
)
with st.sidebar.expander("Show schema given to the model"):
    st.code(pipeline.schema, language="sql")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("Iterative Prompt Refinement for SQL Query Planning via TextGrad")
st.caption(
    "Generate SQL → execute it on SQLite → capture the error, wrong result or slow query plan → an LLM critic "
    "writes a *textual gradient* → an optimiser rewrites the system prompt → the next query is better."
)
if llm_status.mode == "demo":
    st.warning(DEMO_CAVEAT)

if "bench" not in st.session_state:
    st.session_state["bench"] = {}
bench: dict[str, TierBenchmark] = st.session_state["bench"]

tab_live, tab_bench, tab_about = st.tabs(["🔍 Live refinement", "📊 Benchmark", "ℹ️ How it works"])

# --------------------------------------------------------------------------- #
# Tab 1: live refinement
# --------------------------------------------------------------------------- #


def render_round(rnd: RefinementRound) -> None:
    """Show one generate → execute → plan → feedback → refine round."""
    att = rnd.attempt
    with st.container(border=True):
        st.markdown(f"#### Round {rnd.index}: {att.status}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Generated SQL**" if rnd.index > 1 else "**Initial SQL**")
            st.code(att.sql or "(empty)", language="sql")
        with right:
            st.markdown("**Execution result**")
            if not att.result.ok:
                st.error(att.result.error)
            else:
                if att.result.row_count:
                    show_table(att.result.to_dataframe())
                else:
                    st.info("The query ran but returned 0 rows.")
                if att.result.truncated:
                    st.caption("Result truncated to the first 1000 rows.")
                if att.correct is False:
                    st.warning(f"Wrong result: {att.mismatch_reason}.")
                    if att.expected is not None and att.expected.ok:
                        with st.expander("Expected result"):
                            show_table(att.expected.to_dataframe())
                elif att.inefficient:
                    st.warning("Correct result, but the query plan is inefficient (see the plan below).")
                elif att.correct:
                    st.success("Matches the expected result.")
        if att.plan is not None and att.result.ok:
            with st.expander("Query plan & cost", expanded=att.inefficient):
                st.code(att.plan.text(), language="text")
                cost = f"Work: {att.result.vm_steps:,} SQLite VM steps"
                if att.efficiency_ratio is not None and att.expected is not None:
                    cost += f" · reference needs {att.expected.vm_steps:,} ({att.efficiency_ratio:.1f}x ratio)"
                st.caption(cost)
                for issue in att.plan.issues:
                    st.warning(issue)
        if rnd.gradient:
            st.markdown("**TextGrad prompt feedback (textual gradient)**")
            st.info(rnd.gradient)
        if rnd.refined_prompt:
            with st.expander(f"Refined prompt after round {rnd.index}", expanded=True):
                diff = prompt_diff(rnd.prompt_used, rnd.refined_prompt)
                if diff:
                    st.markdown("Changes against the previous prompt:")
                    st.code(diff, language="diff")
                st.markdown("Full refined prompt:")
                st.code(rnd.refined_prompt, language="text")
        elif rnd.gradient:
            st.caption(f"Prompt not changed: {rnd.note}")


with tab_live:
    if is_builtin:
        st.subheader(f"Ask a question  ·  tier: {tier_label}")
        st.caption(TIER_DESCRIPTIONS[tier])
        samples = [i.question for i in items_for(tier)]
        if st.session_state.get("_active_tier") != tier:  # tier switched: start from that tier's first sample
            st.session_state["_active_tier"] = tier
            st.session_state["question"] = samples[0]
            st.session_state.pop("sample_pick", None)

        def _on_pick() -> None:
            picked = st.session_state.get("sample_pick")
            if picked and picked != CUSTOM_OPTION:
                st.session_state["question"] = picked

        st.selectbox("Sample questions for this tier", [CUSTOM_OPTION] + samples, index=1,
                     key="sample_pick", on_change=_on_pick)
        question_key = "question"
        help_text = "In demo mode only the built-in sample questions can be answered."
    else:
        st.subheader("Ask a question about your database")
        st.caption("Without a reference query only execution errors and obvious anti-patterns count as failures. "
                   "Add a reference SQL below to also check correctness and efficiency.")
        st.session_state.setdefault("question_custom", "")
        question_key = "question_custom"
        help_text = "Demo mode cannot answer questions about your own data; configure a model for that."

    question = st.text_area("Natural-language question", key=question_key, height=80, help=help_text)

    st.session_state.setdefault("ref_sql", "")
    with st.expander("Optional: reference SQL (enables correctness and efficiency checks)"):
        ref_sql = st.text_area("Reference SQL", key="ref_sql", height=100,
                               help="For benchmark questions the built-in reference is used automatically.")

    start_options = ["Zero-shot prompt (default)", "Few-shot prompt"]
    if tier is not None and tier in bench:
        start_options.append("TextGrad-optimised prompt (from the Benchmark tab)")
    start_choice = st.radio("Starting prompt", start_options, horizontal=True)
    start_prompt = {0: ZERO_SHOT_PROMPT, 1: FEW_SHOT_PROMPT}.get(start_options.index(start_choice))
    if start_prompt is None:
        start_prompt = bench[tier].textgrad.prompt

    with st.expander("Show the starting system prompt"):
        st.code(start_prompt, language="text")

    if st.button("▶ Generate & refine", type="primary", disabled=not question.strip()):
        rounds: list[RefinementRound] = []
        try:
            with st.spinner("Calling the model…"):
                for rnd in pipeline.refine_question(
                    question.strip(), prompt=start_prompt, reference_sql=ref_sql.strip() or None, max_rounds=max_rounds
                ):
                    rounds.append(rnd)
                    render_round(rnd)
        except LLMError as exc:
            st.error(f"{exc}")
        if rounds:
            first, last = rounds[0].attempt, rounds[-1].attempt
            st.markdown("##### Static prompt vs refined prompt for this question")
            m1, m2, m3 = st.columns(3)
            m1.metric("Static prompt (round 1)", "✅ pass" if first.succeeded else "❌ fail")
            m2.metric("After TextGrad refinement", "✅ pass" if last.succeeded else "❌ fail",
                      delta=("fixed" if (last.succeeded and not first.succeeded) else None))
            m3.metric("Rounds used", len(rounds))
            if last.correct is None:
                st.caption("No reference answer for this question, so correctness was not checked.")

# --------------------------------------------------------------------------- #
# Tab 2: benchmark
# --------------------------------------------------------------------------- #


def show_ci_chart(metric_key: str, benchmarks: list[TierBenchmark]) -> None:
    """Grouped bars (strategy per tier) with 95% confidence-interval whiskers; plain bars as a fallback."""
    frame = ci_frame(benchmarks, metric_key)
    if alt is not None:
        try:
            bars = alt.Chart(frame).mark_bar().encode(
                x=alt.X("Tier:N", title=None, sort=None),
                xOffset=alt.XOffset("Strategy:N"),
                y=alt.Y("Value:Q", title="% of test questions", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Strategy:N", sort=None),
                tooltip=["Tier", "Strategy", alt.Tooltip("Value:Q", format=".0f"),
                         alt.Tooltip("Low:Q", format=".0f"), alt.Tooltip("High:Q", format=".0f")],
            )
            whiskers = alt.Chart(frame).mark_rule().encode(
                x=alt.X("Tier:N", sort=None), xOffset=alt.XOffset("Strategy:N"), y="Low:Q", y2="High:Q"
            )
            chart = bars + whiskers
            for kwargs in (STRETCH, {"use_container_width": True}, {}):
                try:
                    st.altair_chart(chart, **kwargs)
                    return
                except TypeError:
                    continue
        except Exception:
            pass
    pivot = frame.pivot(index="Tier", columns="Strategy", values="Value")[[s.name for s in benchmarks[0].strategies]]
    try:
        st.bar_chart(pivot, stack=False, **CHART_STRETCH)
    except TypeError:
        st.bar_chart(pivot)


def render_overall(benchmarks: list[TierBenchmark]) -> None:
    """Headline numbers pooled over every tier that was run."""
    n = sum(bm.zero_shot.test.n for bm in benchmarks)
    z = sum(bm.zero_shot.test.n_pass for bm in benchmarks)
    f = sum(bm.few_shot.test.n_pass for bm in benchmarks)
    t = sum(bm.textgrad.test.n_pass for bm in benchmarks)
    best = max(z, f)
    c1, c2, c3 = st.columns(3)
    c1.metric("Zero-shot pass rate", pct(z / n))
    c2.metric("Few-shot pass rate", pct(f / n))
    c3.metric("TextGrad pass rate", pct(t / n), delta=f"{(t - best) / n * 100:+.0f} pp vs best static")
    st.caption(f"Pooled over {len(benchmarks)} tier(s), {n} held-out test questions. TextGrad 95% CI: {format_ci(t, n)}.")


def render_tier_results(bm: TierBenchmark) -> None:
    z, f, t = bm.zero_shot.test, bm.few_shot.test, bm.textgrad.test
    base = bm.best_static.test
    st.subheader(f"{bm.label}  ·  held-out test questions (n = {t.n})")
    st.caption(TIER_DESCRIPTIONS[bm.tier])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zero-shot pass rate", pct(z.pass_rate))
    c2.metric("Few-shot pass rate", pct(f.pass_rate))
    c3.metric("TextGrad pass rate", pct(t.pass_rate), delta=f"{(t.pass_rate - base.pass_rate) * 100:+.0f} pp vs best static")
    c4.metric("TextGrad exec. success", pct(t.exec_success),
              delta=f"{(t.exec_success - base.exec_success) * 100:+.0f} pp vs best static")
    st.caption(f"TextGrad pass rate 95% CI: {format_ci(t.n_pass, t.n)}")
    if len(bm.runs) > 1:
        mean, sd, runs = seed_summary(bm)
        st.caption(f"Across {runs} optimisation runs the TextGrad test pass rate was {mean:.0f}% ± {sd:.0f} "
                   f"({', '.join(f'{100 * v:.0f}%' for v in bm.seed_pass_rates)}). The run with the best TRAIN score is shown.")

    opt = bm.optimization
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Training curve** (train questions, after each optimiser step)")
        curve = pd.DataFrame(opt.trajectory, columns=["step", "Pass rate %", "Accuracy %", "Execution success %"]).set_index("step")
        curve *= 100
        st.line_chart(curve, **CHART_STRETCH)
    with right:
        st.markdown("**Per-question outcome**")
        show_table(per_question_frame([bm]).drop(columns=["Tier"]))

    with st.expander(f"Optimiser steps ({len(opt.steps)})"):
        previous = opt.initial_prompt
        for s in opt.steps:
            verdict = "✅ accepted" if s.accepted else ("⏭ skipped" if s.candidate_prompt is None else "❌ rejected")
            st.markdown(f"**Step {s.step}** (epoch {s.epoch}, batch {', '.join(s.batch_ids)}): {verdict}. {s.note}")
            for g in s.gradients:
                st.info(g)
            if s.candidate_prompt:
                st.code(prompt_diff(previous, s.candidate_prompt) or "(no change)", language="diff")
                if s.accepted:
                    previous = s.candidate_prompt
    with st.expander("Final TextGrad-optimised prompt"):
        st.code(bm.textgrad.prompt, language="text")


with tab_bench:
    if not is_builtin:
        st.info("The benchmark (60 labelled questions with reference SQL) runs on the built-in e-commerce database. "
                "Switch the data source back in the sidebar to use it. Your uploaded database works in the "
                "**Live refinement** tab.")
    else:
        st.markdown(
            "Compares three prompting strategies on **held-out test questions** the optimiser never saw: a static "
            "zero-shot prompt, a static few-shot prompt, and the zero-shot prompt optimised with TextGrad on the "
            "**train** questions. A candidate prompt is only kept if it improves the train score. "
            "*Pass rate* = correct result **and** an efficient query plan."
        )
        scope = st.radio("Run benchmark for", [f"Selected tier ({tier_label})", "All three tiers"], horizontal=True)
        tiers_to_run = [tier] if scope.startswith("Selected") else list(TIER_LABELS)
        n_tr, n_te = SIZES[size_choice]
        if llm_status.mode == "live":
            est = len(tiers_to_run) * estimate_calls(n_tr, n_te, epochs, batch_size, n_seeds)
            st.caption(f"Up to ~{est} model calls ({len(tiers_to_run)} tier(s), {size_choice} size). Calls already in the "
                       "response cache are free, and an interrupted run resumes where it stopped.")

        if st.button("▶ Run benchmark", type="primary"):
            bar = st.progress(0.0, text="Starting…")
            state = {"frac": 0.0}

            def on_progress(message: str, fraction: float | None) -> None:
                if fraction is not None:
                    state["frac"] = min(max(fraction, 0.0), 1.0)
                bar.progress(state["frac"], text=message)

            try:
                for t_ in tiers_to_run:
                    state["frac"] = 0.0
                    bench[t_] = pipeline.run_benchmark(
                        t_, epochs=epochs, batch_size=batch_size, seeds=list(range(n_seeds)),
                        size=size_choice, progress=on_progress,
                    )
                bar.progress(1.0, text="Done")
                st.session_state["bench_meta"] = {
                    "name": llm_status.name, "simulated": llm_status.mode == "demo",
                    "settings": {"size": size_choice, "epochs": epochs, "batch size": batch_size,
                                 "seeds": n_seeds, "plan-aware": bool(plan_aware)},
                }
            except LLMError as exc:
                st.error(f"{exc}  Press *Run benchmark* again: completed calls are cached and will not be repeated.")

        if not bench:
            st.info("No results yet. Press **Run benchmark**.")
        else:
            ordered = [bench[t_] for t_ in TIER_LABELS if t_ in bench]
            meta = st.session_state.get("bench_meta", {"name": llm_status.name, "simulated": llm_status.mode == "demo",
                                                        "settings": {}})
            st.caption(f"Results produced with: {meta['name']}")
            if meta["simulated"]:
                st.warning(DEMO_CAVEAT)

            st.markdown("### Headline")
            render_overall(ordered)

            st.markdown("### Summary")
            show_table(benchmark_frame(ordered))
            metric_label = st.radio(
                "Chart metric",
                ["Pass rate (correct & efficient)", "Execution accuracy", "Execution success"],
                horizontal=True,
            )
            metric_key = {"Pass rate (correct & efficient)": "pass_rate", "Execution accuracy": "accuracy",
                          "Execution success": "exec"}[metric_label]
            show_ci_chart(metric_key, ordered)
            st.caption("Whiskers are 95% Wilson confidence intervals. With few test questions they are wide, so "
                       "overlapping whiskers mean the difference is not conclusive.")

            st.markdown("### Is the gain real?")
            show_table(paired_frame(ordered))
            st.caption("TextGrad versus the stronger static prompt on the same test questions. *Fixed* = right now, "
                       "wrong before; *Broken* = the opposite. p-value: exact two-sided sign test (below 0.05 is "
                       "conventionally significant).")

            st.markdown("### Why queries failed")
            tax = taxonomy_frame(ordered)
            if tax.empty or tax.to_numpy().sum() == 0:
                st.success("No failures in any strategy.")
            else:
                st.bar_chart(tax, **CHART_STRETCH)
                with st.expander("Failure counts as a table"):
                    st.dataframe(tax, **STRETCH)

            st.markdown("### Download")
            d1, d2, d3 = st.columns(3)
            d1.download_button(
                "⬇ Report (Markdown)", build_markdown_report(ordered, meta["name"], meta["simulated"], meta["settings"]),
                file_name="textgrad_sql_report.md", mime="text/markdown",
            )
            d2.download_button(
                "⬇ Results (JSON)", build_json_report(ordered, meta["name"], meta["simulated"], meta["settings"]),
                file_name="textgrad_sql_results.json", mime="application/json",
            )
            d3.download_button(
                "⬇ Learned prompts (text)",
                "\n\n".join(f"### {bm.label}\n{bm.textgrad.prompt}" for bm in ordered),
                file_name="textgrad_sql_prompts.txt", mime="text/plain",
            )

            for bm in ordered:
                st.divider()
                render_tier_results(bm)

# --------------------------------------------------------------------------- #
# Tab 3: how it works
# --------------------------------------------------------------------------- #

with tab_about:
    st.markdown(
        """
### The loop

1. **Forward pass:** a system prompt + the schema + a question go to the LLM, which writes SQL.
2. **Execution:** the SQL runs on a read-only SQLite connection. Errors are captured verbatim.
3. **Query planning:** `EXPLAIN QUERY PLAN` is read and the work is measured in SQLite *VM steps*
   (a deterministic cost measure, unlike wall-clock time). A correct query that needs several times the
   work of the reference, for example a correlated subquery re-scanning a table per row, counts as a failure.
4. **Loss:** the failure is described in words: the database error, how the result differs from the expected one,
   or the plan and its cost.
5. **Backward pass:** a critic LLM turns that loss into a *textual gradient*: feedback on how the **prompt**
   should change.
6. **Optimiser step:** another LLM call folds the gradients of a mini-batch into an improved prompt.
7. **Validation gate:** the candidate prompt is kept only if the train score strictly improves.

### The metrics

| Metric | Meaning |
|---|---|
| Execution success | the query ran without a database error |
| Execution accuracy | the result equals the reference query's result (any column order, rows as a multiset) |
| Pass rate | correct **and** not an inefficient plan; the headline metric |
| 95% CI | Wilson interval over the held-out test questions |
| p-value | exact sign test of TextGrad vs the best static prompt on the same questions |

### Why you can trust the comparison

* The optimiser only ever sees **train** questions; every number above is measured on **test** questions.
* The stronger of the two static baselines is used for the significance test.
* With several optimisation runs the headline run is chosen by its **train** score, never by its test score.
* The benchmark has 60 questions in 3 tiers; the confidence intervals show how much (or little) each tier can prove.

### Modes

* **Demo mode** uses a scripted simulator: it demonstrates the machinery, not real model quality.
* **Live mode** uses any OpenAI-compatible model (OpenAI, Ollama, Groq, Gemini, vLLM, ...), configured with
  `API_KEY`, `MODEL` and `BASE_URL`. Responses are cached on disk, so runs are resumable.
* **Your own database:** upload any SQLite file in the sidebar and refine questions about your own data.
"""
    )
