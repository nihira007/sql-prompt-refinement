"""
query_plan.py
=============

Turns SQLite's ``EXPLAIN QUERY PLAN`` output into something a person -- and the
TextGrad critic -- can reason about.

The plan is the "query planning" half of this project: a query can return the
right rows and still be a bad plan (a correlated subquery re-scanning a table for
every outer row, a full table scan, temporary sorting structures ...). Those
findings are written in plain language and become part of the loss that is
back-propagated to the prompt.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PlanStep:
    id: int
    parent: int
    detail: str


@dataclass
class PlanReport:
    """Parsed query plan plus human-readable findings."""

    steps: list[PlanStep] = field(default_factory=list)
    full_scans: list[str] = field(default_factory=list)  # tables/aliases read completely, without an index
    index_lookups: int = 0
    temp_btrees: int = 0
    correlated_subqueries: int = 0
    issues: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def severe(self) -> bool:
        """A pattern that is almost always avoidable: a correlated subquery."""
        return self.correlated_subqueries > 0

    def text(self) -> str:
        """The plan as an indented tree."""
        if self.error:
            return f"(plan unavailable: {self.error})"
        depth: dict[int, int] = {}
        lines = []
        for step in self.steps:
            level = depth.get(step.parent, -1) + 1
            depth[step.id] = level
            lines.append("  " * level + step.detail)
        return "\n".join(lines) if lines else "(empty plan)"


def fetch_plan_steps(sql: str, db_path: str | os.PathLike) -> list[PlanStep]:
    """Run ``EXPLAIN QUERY PLAN`` on a read-only connection (raises sqlite3.Error on bad SQL)."""
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = conn.execute("EXPLAIN QUERY PLAN " + sql.strip().rstrip(";")).fetchall()
        return [PlanStep(int(r[0]), int(r[1]), str(r[3])) for r in rows]
    finally:
        conn.close()


def analyze_plan(sql: str, db_path: str | os.PathLike) -> PlanReport:
    """Fetch and analyse the plan of ``sql``. Never raises."""
    try:
        steps = fetch_plan_steps(sql, db_path)
    except sqlite3.Error as exc:
        return PlanReport(error=str(exc))

    report = PlanReport(steps=steps)
    for step in steps:
        d = step.detail
        upper = d.upper()
        if upper.startswith("SCAN "):
            target = d[5:].split(" USING ")[0].split(" AS ")[0].strip()
            using_index = " USING " in upper
            if not using_index and not target.upper().startswith(("SUBQUERY", "CONSTANT ROW")):
                report.full_scans.append(target)
        if upper.startswith("SEARCH ") or (upper.startswith("SCAN ") and "USING" in upper and "INDEX" in upper):
            report.index_lookups += 1
        if "USE TEMP B-TREE" in upper:
            report.temp_btrees += 1
        if "CORRELATED" in upper:
            report.correlated_subqueries += 1

    if report.correlated_subqueries:
        report.issues.append(
            f"{report.correlated_subqueries} correlated subquery(ies): the inner query is re-executed for every "
            "row of the outer query. Rewrite with a JOIN plus GROUP BY, or a CTE/derived table computed once."
        )
    if len(report.full_scans) >= 3:
        report.issues.append(
            f"{len(report.full_scans)} full table scans ({', '.join(report.full_scans)}): check that joins use "
            "indexed key columns and that tables are not read more often than necessary."
        )
    if report.temp_btrees >= 3:
        report.issues.append(f"{report.temp_btrees} temporary sort/group structures; simplify GROUP BY / ORDER BY / DISTINCT.")
    return report
