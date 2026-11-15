"""
report.py
=========

Turns benchmark results into files a person can keep or share:

* ``build_markdown_report``  -- readable report (tables, confidence intervals, significance,
                                failure taxonomy, learned prompts, per-question results)
* ``build_json_report``      -- the same numbers as machine-readable JSON
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Sequence

from pipeline import (
    TierBenchmark,
    benchmark_frame,
    paired_frame,
    per_question_frame,
    prompt_diff,
    seed_summary,
    taxonomy_frame,
)

DEMO_NOTE = (
    "> **Demo mode.** These results come from an offline simulator that is scripted to make typical "
    "text-to-SQL mistakes. They show that the pipeline works, **not** how a real model performs."
)


def _md_table(df, index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependencies)."""
    frame = df.reset_index() if index else df
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |")
    return "\n".join(lines)


def build_markdown_report(
    benchmarks: Sequence[TierBenchmark], llm_name: str, simulated: bool, settings: dict | None = None
) -> str:
    """Full Markdown report for the given tier benchmarks."""
    settings = settings or {}
    out = [
        "# TextGrad SQL prompt refinement: benchmark report",
        "",
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} with **{llm_name}***",
        "",
    ]
    if simulated:
        out += [DEMO_NOTE, ""]
    if settings:
        out += ["**Settings:** " + ", ".join(f"{k} = {v}" for k, v in settings.items()), ""]

    out += [
        "## Summary",
        "",
        "*Pass rate* = correct result **and** not an inefficient query plan (when plan checking is on). "
        "Intervals are 95% Wilson confidence intervals over the held-out test questions.",
        "",
        _md_table(benchmark_frame(benchmarks)),
        "",
        "## TextGrad versus the best static prompt (test questions)",
        "",
        "*Fixed* = questions TextGrad gets right and the baseline wrong; *Broken* = the opposite. "
        "The p-value is an exact two-sided sign test.",
        "",
        _md_table(paired_frame(benchmarks)),
        "",
    ]

    seeded = [bm for bm in benchmarks if len(bm.runs) > 1]
    if seeded:
        out += ["## Stability across seeds (TextGrad test pass rate)", ""]
        for bm in seeded:
            mean, sd, n = seed_summary(bm)
            runs = ", ".join(f"{100 * v:.0f}%" for v in bm.seed_pass_rates)
            out.append(f"- **{bm.label}**: {mean:.0f}% ± {sd:.0f} over {n} runs ({runs})")
        out.append("")

    out += ["## Why queries failed (all train + test questions)", "", _md_table(taxonomy_frame(benchmarks), index=True), ""]

    for bm in benchmarks:
        out += [f"## {bm.label}: optimisation", ""]
        for s in bm.optimization.steps:
            verdict = "accepted" if s.accepted else ("skipped" if s.candidate_prompt is None else "rejected")
            out.append(f"- Step {s.step} (batch {', '.join(s.batch_ids)}): **{verdict}**. {s.note}")
        out += [
            "",
            "Changes learned by the final prompt:",
            "",
            "```diff",
            prompt_diff(bm.optimization.initial_prompt, bm.textgrad.prompt) or "(no change)",
            "```",
            "",
            "Final prompt:",
            "",
            "```text",
            bm.textgrad.prompt,
            "```",
            "",
        ]

    out += ["## Per-question results", "", _md_table(per_question_frame(benchmarks)), ""]
    return "\n".join(out)


def build_json_report(
    benchmarks: Sequence[TierBenchmark], llm_name: str, simulated: bool, settings: dict | None = None
) -> str:
    """Machine-readable results."""
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": llm_name,
        "simulated": simulated,
        "settings": settings or {},
        "summary": benchmark_frame(benchmarks).to_dict(orient="records"),
        "paired_vs_best_static": paired_frame(benchmarks).to_dict(orient="records"),
        "failure_taxonomy": taxonomy_frame(benchmarks).to_dict(orient="index"),
        "tiers": {
            bm.label: {
                "seed_test_pass_rates": bm.seed_pass_rates,
                "final_prompt": bm.textgrad.prompt,
                "steps": [
                    {"step": s.step, "batch": s.batch_ids, "accepted": s.accepted, "note": s.note,
                     "gradients": s.gradients}
                    for s in bm.optimization.steps
                ],
            }
            for bm in benchmarks
        },
    }
    return json.dumps(payload, indent=2, default=str)
