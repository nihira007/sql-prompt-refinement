"""
pipeline.py
===========

The auto-refining, plan-aware text-to-SQL pipeline.

    question + schema --(LLM + system prompt)--> SQL --(SQLite)--> rows | error
                                   ^                       |            |
                                   |                       +--> EXPLAIN QUERY PLAN + cost
                                   |                                    |
                                   +--- TextGrad: critic feedback <-----+
                                        -> optimiser rewrites the prompt

A query "succeeds" only if it executes, returns the right result AND is not an
inefficient plan (the last check can be switched off). Offered on top of the loop:

* ``refine_question``  -- live, per-question refinement (generator, so a UI can show every
                          round as it happens)
* ``optimize_prompt``  -- mini-batch TextGrad training of the prompt on the *train* questions
                          of a complexity tier; a candidate prompt is accepted only if it
                          measurably helps
* ``run_benchmark``    -- static zero-shot vs static few-shot vs TextGrad-optimised prompt
                          (optionally repeated over several seeds), scored on train AND
                          held-out test questions, per complexity tier
"""

from __future__ import annotations

import difflib
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

import database as db
from benchmark import SIZES, TIER_LABELS, BenchmarkItem, find_item, items_for
from llm_client import BaseLLM
from query_plan import PlanReport, analyze_plan
from stats import format_ci, mean_sd, paired_counts, sign_test, wilson_interval
from textgrad_optimizer import FailureContext, PromptVariable, TextGradOptimizer

ProgressFn = Callable[[str, "float | None"], None]

# --------------------------------------------------------------------------- #
# Static baseline prompts
# --------------------------------------------------------------------------- #

ZERO_SHOT_PROMPT = """\
You are an expert data analyst who writes SQL for SQLite.
Given a database schema and a question in plain English, write one SQL query that answers the question.
Return only the SQL query."""

# Generic worked examples. None of them is a benchmark question.
FEW_SHOT_PROMPT = (
    ZERO_SHOT_PROMPT
    + """

Examples:

Question: List the names of the 5 cheapest products.
SQL: SELECT name FROM products ORDER BY price ASC LIMIT 5;

Question: Show each order id together with the name of the customer who placed it.
SQL: SELECT o.order_id, c.name FROM orders o JOIN customers c ON o.customer_id = c.customer_id;

Question: What is the total number of products?
SQL: SELECT COUNT(*) FROM products;"""
)

# Appended (unchanged) to whatever prompt is being optimised, so that the
# optimiser can never "optimise away" the output format.
OUTPUT_FOOTER = (
    "Respond with exactly one read-only SQLite SELECT statement inside a ```sql fenced block. "
    "Do not add explanations."
)


def extract_sql(text: str) -> str:
    """Pull the SQL statement out of a model reply (fenced block, or bare SELECT/WITH)."""
    text = text or ""
    fenced = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    bare = re.search(r"\b(select|with)\b.*", text, re.DOTALL | re.IGNORECASE)
    return bare.group(0).strip() if bare else text.strip()


def prompt_diff(old: str, new: str) -> str:
    """Unified diff between two prompts (empty string if identical)."""
    diff = difflib.unified_diff(
        old.splitlines(), new.splitlines(), fromfile="previous prompt", tofile="refined prompt", lineterm="", n=2
    )
    return "\n".join(diff)


def estimate_calls(n_train: int, n_test: int, epochs: int, batch_size: int, seeds: int = 1) -> int:
    """Rough upper bound of model calls for ONE tier (cached calls are free)."""
    steps = epochs * math.ceil(n_train / max(batch_size, 1))
    per_step = batch_size + 1 + n_train  # gradients + optimiser + validation of the candidate
    static = 2 * (n_train + n_test)
    return static + seeds * (steps * per_step + n_test)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class Attempt:
    """One pass of generate -> execute -> plan -> evaluate."""

    question: str
    sql: str
    result: db.QueryResult
    expected: db.QueryResult | None = None  # result of the reference SQL, if there is one
    correct: bool | None = None  # None when no reference is available
    mismatch_reason: str = ""
    plan: PlanReport | None = None
    efficiency_ratio: float | None = None  # VM steps vs reference (None without reference)
    inefficient: bool = False

    @property
    def executed(self) -> bool:
        return self.result.ok

    @property
    def failure_type(self) -> str | None:
        if not self.result.ok:
            return "execution_error"
        if self.correct is False:
            return "result_mismatch"
        if self.inefficient:
            return "inefficient_plan"
        return None

    @property
    def succeeded(self) -> bool:
        return self.failure_type is None

    @property
    def status(self) -> str:
        if not self.result.ok:
            return "❌ error"
        if self.correct is False:
            return "⚠️ wrong result"
        if self.inefficient:
            return f"🐢 slow ({self.efficiency_ratio:.1f}x)" if self.efficiency_ratio else "🐢 slow plan"
        return "✅ correct" if self.correct else "✅ executed"


def failure_category(attempt: Attempt) -> str | None:
    """Coarse taxonomy of why an attempt failed (None if it succeeded)."""
    kind = attempt.failure_type
    if kind is None:
        return None
    if kind == "result_mismatch":
        return "Wrong result"
    if kind == "inefficient_plan":
        return "Inefficient plan"
    err = (attempt.result.error or "").lower()
    for needle, label in (
        ("no such function", "Unknown function"),
        ("no such column", "Unknown column"),
        ("no such table", "Unknown table"),
        ("ambiguous column", "Ambiguous column"),
        ("misuse of", "Aggregate/window misuse"),
        ("syntax error", "Syntax error"),
        ("time limit", "Timeout"),
        ("rejected", "Not a SELECT"),
        ("empty sql", "Not a SELECT"),
    ):
        if needle in err:
            return label
    return "Other error"


FAILURE_CATEGORIES = [
    "Syntax error", "Unknown function", "Unknown column", "Unknown table", "Ambiguous column",
    "Aggregate/window misuse", "Timeout", "Not a SELECT", "Other error", "Wrong result", "Inefficient plan",
]


@dataclass
class RefinementRound:
    """One iteration of the live loop (see ``AutoRefiningPipeline.refine_question``)."""

    index: int
    prompt_used: str
    attempt: Attempt
    gradient: str | None = None  # TextGrad feedback (None if the attempt succeeded)
    refined_prompt: str | None = None  # prompt after the TGD step (None if none was produced)
    note: str = ""


@dataclass
class EvalReport:
    """Scores of one prompt over a list of benchmark items."""

    items: list[BenchmarkItem]
    attempts: list[Attempt]

    @property
    def n(self) -> int:
        return len(self.items)

    def _share(self, predicate: Callable[[Attempt], bool]) -> float:
        return sum(predicate(a) for a in self.attempts) / self.n if self.n else 0.0

    @property
    def exec_success(self) -> float:
        """Share of queries that ran without a database error."""
        return self._share(lambda a: a.executed)

    @property
    def accuracy(self) -> float:
        """Execution accuracy: share whose result equals the reference result."""
        return self._share(lambda a: bool(a.correct))

    @property
    def pass_rate(self) -> float:
        """Share that is correct AND not an inefficient plan (the headline metric)."""
        return self._share(lambda a: a.succeeded)

    @property
    def n_pass(self) -> int:
        return sum(a.succeeded for a in self.attempts)

    @property
    def score(self) -> tuple[float, float, float]:
        """Ordering used to accept/reject candidate prompts."""
        return (self.pass_rate, self.accuracy, self.exec_success)

    def flags(self) -> list[bool]:
        return [a.succeeded for a in self.attempts]

    def failed(self) -> list[tuple[BenchmarkItem, Attempt]]:
        return [(i, a) for i, a in zip(self.items, self.attempts) if not a.succeeded]


@dataclass
class StepRecord:
    """Bookkeeping for one optimiser step."""

    step: int
    epoch: int
    batch_ids: list[str]
    n_failures: int
    gradients: list[str] = field(default_factory=list)
    candidate_prompt: str | None = None
    score_before: tuple[float, float, float] = (0.0, 0.0, 0.0)
    score_after: tuple[float, float, float] | None = None
    accepted: bool = False
    note: str = ""


@dataclass
class OptimizationResult:
    initial_prompt: str
    final_prompt: str
    steps: list[StepRecord]
    trajectory: list[tuple[int, float, float, float]]  # (step, pass rate, accuracy, exec success) on train
    final_train: EvalReport
    seed: int = 0


@dataclass
class StrategyScores:
    name: str
    prompt: str
    train: EvalReport
    test: EvalReport


@dataclass
class SeedRun:
    seed: int
    optimization: OptimizationResult
    test: EvalReport


@dataclass
class TierBenchmark:
    tier: str
    zero_shot: StrategyScores
    few_shot: StrategyScores
    textgrad: StrategyScores  # the run with the best TRAIN score (never chosen by test score)
    optimization: OptimizationResult
    runs: list[SeedRun] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0

    @property
    def label(self) -> str:
        return TIER_LABELS[self.tier]

    @property
    def strategies(self) -> list[StrategyScores]:
        return [self.zero_shot, self.few_shot, self.textgrad]

    @property
    def best_static(self) -> StrategyScores:
        """Stronger static baseline on the test set (few-shot wins ties)."""
        a, b = self.zero_shot, self.few_shot
        return a if a.test.score > b.test.score else b

    @property
    def seed_pass_rates(self) -> list[float]:
        return [r.test.pass_rate for r in self.runs]


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


class AutoRefiningPipeline:
    def __init__(
        self,
        llm: BaseLLM,
        db_path: str | None = None,
        max_prompt_chars: int = 3500,
        plan_aware: bool = True,
        efficiency_threshold: float = 3.0,
        min_extra_steps: int = 3000,
    ) -> None:
        self.llm = llm
        builtin = db.init_database()
        self.db_path = Path(db_path) if db_path else builtin
        self.is_builtin = self.db_path.resolve() == builtin.resolve()
        self.schema = db.get_schema_text(None if self.is_builtin else self.db_path)
        self.optimizer = TextGradOptimizer(llm, max_prompt_chars=max_prompt_chars)
        self.plan_aware = plan_aware
        self.efficiency_threshold = efficiency_threshold
        self.min_extra_steps = min_extra_steps
        self._expected_cache: dict[str, db.QueryResult] = {}

    # ---------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------- #
    def generate_sql(self, prompt: str, question: str) -> str:
        """LLM call: system prompt + schema + question -> SQL text."""
        user = (
            f"<SCHEMA>\n{self.schema}\n</SCHEMA>\n"
            f"<QUESTION>\n{question}\n</QUESTION>\n"
            "Write the SQL query."
        )
        system = f"{prompt.strip()}\n\n{OUTPUT_FOOTER}"
        return extract_sql(self.llm.invoke(system, user, role="generate"))

    def _expected(self, reference_sql: str) -> db.QueryResult:
        if reference_sql not in self._expected_cache:
            self._expected_cache[reference_sql] = db.run_sql(reference_sql, self.db_path)
        return self._expected_cache[reference_sql]

    def run_attempt(self, prompt: str, question: str, reference_sql: str | None = None) -> Attempt:
        """Generate SQL, execute it, inspect its plan and (if a reference exists) judge it."""
        sql = self.generate_sql(prompt, question)
        result = db.run_sql(sql, self.db_path)
        attempt = Attempt(question=question, sql=sql, result=result)
        if result.ok:
            attempt.plan = analyze_plan(sql, self.db_path)
        if reference_sql:
            expected = self._expected(reference_sql)
            attempt.expected = expected
            if result.ok:
                attempt.correct, attempt.mismatch_reason = db.results_match(
                    result, expected, ordered=db.has_order_by(reference_sql)
                )
            else:
                attempt.correct, attempt.mismatch_reason = False, "the query failed to execute"
            if result.ok and expected.ok:
                attempt.efficiency_ratio = result.vm_steps / max(expected.vm_steps, 1)
                attempt.inefficient = (
                    self.plan_aware
                    and attempt.correct is True
                    and attempt.efficiency_ratio > self.efficiency_threshold
                    and result.vm_steps - expected.vm_steps >= self.min_extra_steps
                )
        elif result.ok and attempt.plan is not None:
            # No reference to measure against: only flag the clearest anti-pattern.
            attempt.inefficient = (
                self.plan_aware and attempt.plan.severe and result.vm_steps >= self.min_extra_steps
            )
        return attempt

    def evaluate_prompt(
        self, prompt: str, items: Sequence[BenchmarkItem], progress: ProgressFn | None = None, label: str = ""
    ) -> EvalReport:
        """Score ``prompt`` on every item (execution success, accuracy and plan-efficient pass rate)."""
        attempts = []
        for n, item in enumerate(items, start=1):
            if progress:
                progress(f"{label} question {n}/{len(items)}", None)
            attempts.append(self.run_attempt(prompt, item.question, item.gold_sql))
        return EvalReport(list(items), attempts)

    # ---------------------------------------------------------------- #
    # Backward pass helpers
    # ---------------------------------------------------------------- #
    @staticmethod
    def _failure_context(attempt: Attempt, reference_sql: str | None) -> FailureContext:
        kind = attempt.failure_type or "result_mismatch"
        return FailureContext(
            question=attempt.question,
            sql=attempt.sql,
            failure_type=kind,
            error=attempt.result.error,
            mismatch_reason=attempt.mismatch_reason,
            actual_preview=attempt.result.preview() if attempt.result.ok else "",
            expected_preview=attempt.expected.preview() if attempt.expected else "",
            reference_sql=reference_sql,
            plan_text=attempt.plan.text() if attempt.plan and kind == "inefficient_plan" else "",
            plan_issues=list(attempt.plan.issues) if attempt.plan and kind == "inefficient_plan" else [],
            actual_steps=attempt.result.vm_steps,
            expected_steps=attempt.expected.vm_steps if attempt.expected else 0,
        )

    # ---------------------------------------------------------------- #
    # Live per-question refinement
    # ---------------------------------------------------------------- #
    def refine_question(
        self,
        question: str,
        prompt: str | None = None,
        reference_sql: str | None = None,
        max_rounds: int = 3,
    ) -> Iterator[RefinementRound]:
        """
        Generate SQL for ``question``; while it fails, ask TextGrad for feedback,
        rewrite the prompt and try again (at most ``max_rounds`` attempts).

        Yields one ``RefinementRound`` per attempt as soon as it is complete.
        On the built-in database, a benchmark question is judged against its gold SQL
        (wrong-but-executable and slow queries are caught too). Otherwise, unless a
        ``reference_sql`` is given, only execution errors and obvious anti-patterns
        (correlated subqueries) count as failures.
        """
        if reference_sql is None and self.is_builtin:
            known = find_item(question)
            reference_sql = known.gold_sql if known else None

        variable = PromptVariable(prompt if prompt is not None else ZERO_SHOT_PROMPT)
        for index in range(1, max_rounds + 1):
            used = variable.value
            attempt = self.run_attempt(used, question, reference_sql)
            rnd = RefinementRound(index=index, prompt_used=used, attempt=attempt)
            if attempt.succeeded:
                rnd.note = "The query executed" + (" and matches the expected result." if attempt.correct else ".")
                yield rnd
                return

            variable.zero_grad()
            rnd.gradient = self.optimizer.compute_gradient(
                variable, self._failure_context(attempt, reference_sql), self.schema
            )
            proposal = self.optimizer.propose_update(variable)
            rnd.note = proposal.reason
            if proposal.valid and proposal.changed and proposal.new_value:
                variable.update(proposal.new_value)
                rnd.refined_prompt = proposal.new_value
            yield rnd
            if rnd.refined_prompt is None:  # optimiser could not improve the prompt: stop early
                return

    # ---------------------------------------------------------------- #
    # Mini-batch TextGrad training
    # ---------------------------------------------------------------- #
    def optimize_prompt(
        self,
        base_prompt: str,
        train_items: Sequence[BenchmarkItem],
        epochs: int = 2,
        batch_size: int = 3,
        seed: int = 0,
        progress: ProgressFn | None = None,
        initial_eval: EvalReport | None = None,
    ) -> OptimizationResult:
        """
        Optimise ``base_prompt`` with validation-gated textual gradient descent.

        For every mini-batch: collect the failing questions, turn each failure into a
        textual gradient, let the optimiser fold all gradients into ONE candidate
        prompt, re-score the candidate on the whole train set and keep it only if the
        score (pass rate, then accuracy, then execution success) strictly improves.
        Stops early at a 100% train pass rate.
        """
        rng = random.Random(seed)
        variable = PromptVariable(base_prompt)
        train_items = list(train_items)
        current = initial_eval or self.evaluate_prompt(variable.value, train_items, progress, "Scoring start prompt:")
        trajectory = [(0, current.pass_rate, current.accuracy, current.exec_success)]
        steps: list[StepRecord] = []
        total_steps = epochs * -(-len(train_items) // max(batch_size, 1))
        step_no = 0

        def snapshot() -> tuple[int, float, float, float]:
            return (step_no, current.pass_rate, current.accuracy, current.exec_success)

        for epoch in range(1, epochs + 1):
            order = list(train_items)
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                if current.pass_rate >= 1.0:
                    break
                batch = order[start : start + batch_size]
                batch_ids = [i.id for i in batch]
                step_no += 1
                frac = step_no / max(total_steps, 1)
                fails = [(i, a) for i, a in current.failed() if i.id in batch_ids]
                record = StepRecord(step_no, epoch, batch_ids, len(fails), score_before=current.score)

                if not fails:
                    record.note = "No failures in this batch; nothing to learn."
                    steps.append(record)
                    trajectory.append(snapshot())
                    continue

                if progress:
                    progress(f"Step {step_no}/{total_steps}: computing textual gradients for {len(fails)} failure(s)", frac)
                variable.zero_grad()
                for item, attempt in fails:
                    record.gradients.append(
                        self.optimizer.compute_gradient(
                            variable, self._failure_context(attempt, item.gold_sql), self.schema
                        )
                    )

                if progress:
                    progress(f"Step {step_no}/{total_steps}: rewriting the prompt", frac)
                proposal = self.optimizer.propose_update(variable)
                if not (proposal.valid and proposal.changed and proposal.new_value):
                    record.note = f"Rejected: {proposal.reason}."
                    steps.append(record)
                    trajectory.append(snapshot())
                    continue

                record.candidate_prompt = proposal.new_value
                if progress:
                    progress(f"Step {step_no}/{total_steps}: validating the candidate prompt", frac)
                candidate = self.evaluate_prompt(proposal.new_value, train_items)
                record.score_after = candidate.score
                if candidate.score > current.score:
                    record.accepted = True
                    record.note = (
                        f"Accepted: train pass rate {current.pass_rate:.0%} -> {candidate.pass_rate:.0%}, "
                        f"execution success {current.exec_success:.0%} -> {candidate.exec_success:.0%}."
                    )
                    variable.update(proposal.new_value)
                    current = candidate
                else:
                    record.note = "Rejected: the candidate prompt did not improve the train score."
                steps.append(record)
                trajectory.append(snapshot())
            if current.pass_rate >= 1.0:
                break

        variable.zero_grad()
        return OptimizationResult(
            initial_prompt=base_prompt.strip(),
            final_prompt=variable.value,
            steps=steps,
            trajectory=trajectory,
            final_train=current,
            seed=seed,
        )

    # ---------------------------------------------------------------- #
    # Static vs TextGrad benchmark
    # ---------------------------------------------------------------- #
    def run_benchmark(
        self,
        tier: str,
        epochs: int = 2,
        batch_size: int = 3,
        seeds: Sequence[int] = (0,),
        size: str = "Full",
        progress: ProgressFn | None = None,
    ) -> TierBenchmark:
        """
        Compare, for one complexity tier:
          1. static zero-shot prompt
          2. static few-shot prompt
          3. zero-shot prompt optimised with TextGrad on the TRAIN questions (once per seed)
        Every strategy is scored on the train questions and on held-out TEST questions.
        With several seeds the TextGrad run with the best TRAIN score is the headline result;
        all runs are kept in ``TierBenchmark.runs`` so the spread is visible.
        """
        if not self.is_builtin:
            raise ValueError("The benchmark needs the built-in e-commerce database.")
        n_train, n_test = SIZES[size]
        train, test = items_for(tier, "train", n_train), items_for(tier, "test", n_test)
        tag = TIER_LABELS[tier]

        def evaluate(prompt: str, items: list[BenchmarkItem], what: str) -> EvalReport:
            if progress:
                progress(f"{tag}: scoring {what}", None)
            return self.evaluate_prompt(prompt, items)

        zero_train = evaluate(ZERO_SHOT_PROMPT, train, "zero-shot (train)")
        zero_test = evaluate(ZERO_SHOT_PROMPT, test, "zero-shot (test)")
        few_train = evaluate(FEW_SHOT_PROMPT, train, "few-shot (train)")
        few_test = evaluate(FEW_SHOT_PROMPT, test, "few-shot (test)")

        runs: list[SeedRun] = []
        for k, seed in enumerate(seeds, start=1):
            suffix = f" (seed {seed}, run {k}/{len(seeds)})" if len(seeds) > 1 else ""

            def tier_progress(message: str, fraction: float | None, _s: str = suffix) -> None:
                if progress:
                    progress(f"{tag}{_s}: {message}", fraction)

            opt = self.optimize_prompt(ZERO_SHOT_PROMPT, train, epochs, batch_size, seed, tier_progress, initial_eval=zero_train)
            tg_test = evaluate(opt.final_prompt, test, f"TextGrad-optimised prompt{suffix} (test)")
            runs.append(SeedRun(seed, opt, tg_test))

        chosen = max(runs, key=lambda r: r.optimization.final_train.score)  # first maximum wins ties
        return TierBenchmark(
            tier=tier,
            zero_shot=StrategyScores("Static zero-shot", ZERO_SHOT_PROMPT, zero_train, zero_test),
            few_shot=StrategyScores("Static few-shot", FEW_SHOT_PROMPT, few_train, few_test),
            textgrad=StrategyScores(
                "TextGrad-optimised", chosen.optimization.final_prompt, chosen.optimization.final_train, chosen.test
            ),
            optimization=chosen.optimization,
            runs=runs,
            n_train=len(train),
            n_test=len(test),
        )


# --------------------------------------------------------------------------- #
# Tables for the UI / reports
# --------------------------------------------------------------------------- #


def benchmark_frame(benchmarks: Sequence[TierBenchmark]):
    """One row per (tier, strategy): execution success, accuracy and plan-efficient pass rate in percent."""
    import pandas as pd

    rows = []
    for bm in benchmarks:
        for s in bm.strategies:
            rows.append(
                {
                    "Tier": bm.label,
                    "Strategy": s.name,
                    "Train exec. success %": round(s.train.exec_success * 100, 1),
                    "Train pass rate %": round(s.train.pass_rate * 100, 1),
                    "Test exec. success %": round(s.test.exec_success * 100, 1),
                    "Test accuracy %": round(s.test.accuracy * 100, 1),
                    "Test pass rate %": round(s.test.pass_rate * 100, 1),
                    "Test pass rate (95% CI)": format_ci(s.test.n_pass, s.test.n),
                }
            )
    return pd.DataFrame(rows)


def ci_frame(benchmarks: Sequence[TierBenchmark], metric: str = "pass_rate"):
    """Long table (tier, strategy, value, low, high) for charts with error bars; values in percent."""
    import pandas as pd

    rows = []
    for bm in benchmarks:
        for s in bm.strategies:
            rep = s.test
            k = sum(a.succeeded for a in rep.attempts) if metric == "pass_rate" else (
                sum(bool(a.correct) for a in rep.attempts) if metric == "accuracy" else sum(a.executed for a in rep.attempts)
            )
            lo, hi = wilson_interval(k, rep.n)
            rows.append({"Tier": bm.label, "Strategy": s.name, "Value": 100 * k / max(rep.n, 1),
                         "Low": 100 * lo, "High": 100 * hi})
    return pd.DataFrame(rows)


def per_question_frame(benchmarks: Sequence[TierBenchmark]):
    """One row per benchmark question showing how each strategy did on it."""
    import pandas as pd

    rows = []
    for bm in benchmarks:
        for split in ("train", "test"):
            reports = [getattr(s, split) for s in bm.strategies]
            for idx, item in enumerate(reports[0].items):
                rows.append(
                    {
                        "Tier": bm.label,
                        "Split": split,
                        "ID": item.id,
                        "Question": item.question,
                        "Zero-shot": reports[0].attempts[idx].status,
                        "Few-shot": reports[1].attempts[idx].status,
                        "TextGrad": reports[2].attempts[idx].status,
                    }
                )
    return pd.DataFrame(rows)


def taxonomy_frame(benchmarks: Sequence[TierBenchmark]):
    """Failure categories per strategy (all train + test questions of the given tiers)."""
    import pandas as pd

    counts: dict[str, dict[str, int]] = {}
    for bm in benchmarks:
        for s in bm.strategies:
            row = counts.setdefault(s.name, {})
            for rep in (s.train, s.test):
                for a in rep.attempts:
                    cat = failure_category(a)
                    if cat:
                        row[cat] = row.get(cat, 0) + 1
    names = [s.name for s in benchmarks[0].strategies] if benchmarks else []
    used = [c for c in FAILURE_CATEGORIES if any(c in counts.get(n, {}) for n in names)]
    return pd.DataFrame({c: [counts.get(n, {}).get(c, 0) for n in names] for c in used}, index=names)


def paired_frame(benchmarks: Sequence[TierBenchmark]):
    """TextGrad vs the best static baseline on the TEST questions: fixed / broken questions and sign-test p."""
    import pandas as pd

    rows = []
    all_base: list[bool] = []
    all_tg: list[bool] = []
    for bm in benchmarks:
        base = bm.best_static
        b_flags, t_flags = base.test.flags(), bm.textgrad.test.flags()
        fixed, broken = paired_counts(b_flags, t_flags)
        all_base += b_flags
        all_tg += t_flags
        rows.append({"Tier": bm.label, "Baseline": base.name, "n": len(b_flags), "Fixed": fixed, "Broken": broken,
                     "p-value": round(sign_test(fixed, broken), 4)})
    if len(benchmarks) > 1:
        fixed, broken = paired_counts(all_base, all_tg)
        rows.append({"Tier": "All tiers", "Baseline": "best static per tier", "n": len(all_base), "Fixed": fixed,
                     "Broken": broken, "p-value": round(sign_test(fixed, broken), 4)})
    return pd.DataFrame(rows)


def seed_summary(bm: TierBenchmark) -> tuple[float, float, int]:
    """(mean, sd, runs) of the TextGrad test pass rate over seeds, in percent."""
    mean, sd = mean_sd([100 * v for v in bm.seed_pass_rates])
    return mean, sd, len(bm.runs)
