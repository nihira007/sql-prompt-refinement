"""
test_project.py -- test suite for the whole project.

Run with either of:

    python test_project.py
    pytest -q test_project.py

No API key, network access or Streamlit is needed (the offline simulator plays the LLM).
The tests use a throw-away database file and never touch your ecommerce.db.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import traceback
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="textgrad_sql_tests_")
os.environ["TEXTGRAD_SQL_DB"] = str(Path(_TMP) / "test_ecommerce.db")  # must be set before importing database
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark as bm  # noqa: E402
import database as db  # noqa: E402
import pipeline as pl  # noqa: E402
import query_plan as qp  # noqa: E402
import report  # noqa: E402
import stats  # noqa: E402
from llm_client import BaseLLM, CachedLLM, LLMError, get_llm  # noqa: E402
from mock_llm import FAILURE_CHAINS, MockLLM  # noqa: E402
from textgrad_optimizer import FailureContext, PromptVariable, TextGradOptimizer, extract_improved_prompt  # noqa: E402

M20 = next(i for i in bm.BENCHMARK if i.id == "M20")
S1 = next(i for i in bm.BENCHMARK if i.id == "S1")


def _pipeline(**kw) -> pl.AutoRefiningPipeline:
    return pl.AutoRefiningPipeline(MockLLM(), **kw)


# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #


def test_database_is_deterministic_and_populated():
    path = db.init_database(reset=True)
    assert db.get_table_overview(path) == {"customers": 40, "order_items": 312, "orders": 120, "products": 21}
    first = db.run_sql("SELECT * FROM orders ORDER BY order_id").rows
    db.init_database(reset=True)
    assert db.run_sql("SELECT * FROM orders ORDER BY order_id").rows == first


def test_run_sql_is_read_only_and_single_statement():
    db.init_database()
    for bad in ("DROP TABLE orders", "DELETE FROM orders", "INSERT INTO products VALUES (99,'x','Toys',1)",
                "UPDATE orders SET status='x'", "PRAGMA writable_schema=1", ""):
        assert not db.run_sql(bad).ok, bad
    two = db.run_sql("SELECT 1; DROP TABLE orders")
    assert not two.ok
    assert db.run_sql("SELECT COUNT(*) FROM orders").rows == [(120,)]


def test_run_sql_captures_errors_instead_of_raising():
    res = db.run_sql("SELECT YEAR(order_date) FROM orders")
    assert not res.ok and "no such function" in res.error.lower()
    assert "no such column" in db.run_sql("SELECT nope FROM orders").error.lower()
    assert "syntax error" in db.run_sql("SELECT TOP 3 name FROM products").error.lower()


def test_run_sql_timeout_aborts_runaway_query():
    runaway = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x < 100000000) "
               "SELECT COUNT(*) FROM c")
    res = db.run_sql(runaway, timeout_s=0.3)
    assert not res.ok and "time limit" in res.error


def test_vm_steps_measure_work():
    cheap = db.run_sql("SELECT COUNT(*) FROM customers").vm_steps
    costly = db.run_sql("SELECT COUNT(*) FROM order_items a, order_items b WHERE a.item_id < b.item_id").vm_steps
    assert 0 < cheap < costly
    assert db.run_sql("SELECT COUNT(*) FROM customers").vm_steps == cheap  # deterministic


def test_results_match_rules():
    a = db.run_sql("SELECT name, price FROM products WHERE product_id <= 3")
    swapped = db.run_sql("SELECT price, name FROM products WHERE product_id <= 3")
    assert db.results_match(swapped, a)[0], "column order must not matter"
    rev = db.run_sql("SELECT name, price FROM products WHERE product_id <= 3 ORDER BY product_id DESC")
    assert db.results_match(rev, a)[0], "row order ignored unless ordered"
    assert not db.results_match(rev, a, ordered=True)[0]
    fewer = db.run_sql("SELECT name, price FROM products WHERE product_id <= 2")
    assert not db.results_match(fewer, a)[0]
    assert db.results_match(db.run_sql("SELECT 1.004"), db.run_sql("SELECT 1.0"))[0], "floats compare to 2 decimals"
    assert not db.results_match(db.run_sql("SELECT nope"), a)[0]


def test_custom_database_introspection():
    path = str(Path(_TMP) / "pets.db")
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE pets(id INTEGER PRIMARY KEY, name TEXT); INSERT INTO pets VALUES (1,'Rex');"
                       "CREATE INDEX idx_pets ON pets(name);")
    conn.commit(); conn.close()
    assert db.validate_sqlite_file(path)[0]
    text = db.introspect_schema(path)
    assert "CREATE TABLE pets" in text and "Rex" in text and "CREATE INDEX idx_pets" in text
    assert not db.validate_sqlite_file(__file__)[0]
    assert db.run_sql("SELECT COUNT(*) FROM pets", path).rows == [(1,)]
    assert db.get_schema_text(path) == text and db.get_schema_text() == db.SCHEMA_DDL


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #


def test_benchmark_is_sound():
    assert bm.validate_benchmark() == []
    assert len(bm.BENCHMARK) == 60
    for tier in bm.TIER_LABELS:
        assert len(bm.items_for(tier, "train")) == 12 and len(bm.items_for(tier, "test")) == 8


def test_benchmark_train_and_test_are_disjoint():
    train = {bm.normalize_question(i.question) for i in bm.BENCHMARK if i.split == "train"}
    test = {bm.normalize_question(i.question) for i in bm.BENCHMARK if i.split == "test"}
    assert not train & test


def test_benchmark_lookup_is_typo_tolerant():
    assert bm.find_item("how many customers are from canada").id == "S1"
    assert bm.find_item_fuzzy("How many customers are from Canadaa?").id == "S1"
    assert bm.find_item_fuzzy("completely unrelated question about weather") is None


def test_size_presets_take_leading_items():
    train, test = bm.SIZES["Quick"]
    assert [i.id for i in bm.items_for("single", "train", train)] == ["S1", "S2", "S3", "S4"]
    assert len(bm.items_for("multi", "test", test)) == test


# --------------------------------------------------------------------------- #
# query plan
# --------------------------------------------------------------------------- #


def test_plan_flags_correlated_subquery_but_not_join():
    good = qp.analyze_plan("SELECT p.name, SUM(oi.quantity) FROM products p LEFT JOIN order_items oi "
                           "ON oi.product_id = p.product_id GROUP BY p.product_id", db.get_db_path())
    slow = qp.analyze_plan("SELECT p.name, (SELECT SUM(oi.quantity) FROM order_items oi "
                           "WHERE oi.product_id = p.product_id) FROM products p", db.get_db_path())
    assert not good.severe and slow.severe and slow.issues
    assert "CORRELATED" in slow.text().upper()


def test_plan_handles_invalid_sql():
    report_ = qp.analyze_plan("SELECT nope FROM nowhere", db.get_db_path())
    assert report_.error and not report_.steps


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def test_wilson_interval():
    lo, hi = stats.wilson_interval(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35
    lo, hi = stats.wilson_interval(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.8
    lo, hi = stats.wilson_interval(5, 10)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 0.01
    assert stats.wilson_interval(0, 0) == (0.0, 0.0)


def test_sign_test_and_pairs():
    assert stats.sign_test(0, 0) == 1.0
    assert abs(stats.sign_test(4, 0) - 0.125) < 1e-12
    assert abs(stats.sign_test(8, 0) - 2 / 256) < 1e-12
    assert stats.sign_test(3, 3) == 1.0
    assert stats.paired_counts([True, False, False, True], [True, True, False, False]) == (1, 1)
    mean, sd = stats.mean_sd([1, 2, 3])
    assert mean == 2 and abs(sd - 1.0) < 1e-12 and stats.mean_sd([5]) == (5, 0.0)


# --------------------------------------------------------------------------- #
# TextGrad optimiser
# --------------------------------------------------------------------------- #


class _Scripted(BaseLLM):
    name = "scripted"

    def __init__(self, reply: str) -> None:
        super().__init__()
        self.reply = reply
        self.seen: list[tuple[str, str]] = []

    def _invoke(self, system, user, role):
        self.seen.append((role, user))
        return self.reply


def test_extract_improved_prompt():
    assert extract_improved_prompt("x <IMPROVED_PROMPT>\n hello \n</IMPROVED_PROMPT> y") == "hello"
    assert extract_improved_prompt("no tags") is None
    assert extract_improved_prompt("<IMPROVED_PROMPT>  </IMPROVED_PROMPT>") is None


def test_optimizer_rejects_unusable_updates():
    var = PromptVariable("base prompt")
    assert not TextGradOptimizer(_Scripted("x")).propose_update(var).valid  # no gradients yet
    var.gradients.append("fix it")
    assert not TextGradOptimizer(_Scripted("plain text")).propose_update(var).valid
    assert not TextGradOptimizer(_Scripted(f"<IMPROVED_PROMPT>{'x' * 5000}</IMPROVED_PROMPT>")).propose_update(var).valid
    same = TextGradOptimizer(_Scripted("<IMPROVED_PROMPT>base prompt</IMPROVED_PROMPT>")).propose_update(var)
    assert same.valid and not same.changed
    new = TextGradOptimizer(_Scripted("<IMPROVED_PROMPT>better</IMPROVED_PROMPT>")).propose_update(var)
    assert new.valid and new.changed and new.new_value == "better"
    assert var.value == "base prompt", "propose_update must not modify the variable"


def test_gradient_contains_the_loss_including_plan_and_cost():
    llm = _Scripted("feedback")
    var = PromptVariable("p")
    ctx = FailureContext(question="q", sql="SELECT 1", failure_type="inefficient_plan", plan_text="SCAN p",
                         plan_issues=["correlated"], actual_steps=30000, expected_steps=6000)
    TextGradOptimizer(llm).compute_gradient(var, ctx, "schema")
    sent = llm.seen[-1][1]
    assert "inefficient" in sent and "SCAN p" in sent and "5.0x" in sent and var.gradients == ["feedback"]
    err = FailureContext(question="q", sql="s", failure_type="execution_error", error="no such function: YEAR")
    assert "no such function: YEAR" in err.loss_description()


# --------------------------------------------------------------------------- #
# LLM client
# --------------------------------------------------------------------------- #


def test_cached_llm_answers_repeats_from_disk_and_survives_restart():
    inner = _Scripted("answer")
    path = Path(_TMP) / "cache.sqlite"
    cached = CachedLLM(inner, path)
    assert cached.invoke("sys", "usr") == "answer" and cached.invoke("sys", "usr") == "answer"
    assert len(inner.seen) == 1 and cached.hits == 1 and cached.misses == 1
    assert cached.invoke("sys", "different") == "answer" and len(inner.seen) == 2
    reopened = CachedLLM(_Scripted("other"), path)  # same name => same keys
    assert reopened.invoke("sys", "usr") == "answer" and not reopened.inner.seen
    assert reopened.clear_cache() == 2 and reopened.entries() == 0


def test_cached_llm_never_caches_errors_or_empty_answers():
    class Flaky(BaseLLM):
        name = "flaky"
        calls = 0

        def _invoke(self, system, user, role):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise LLMError("boom")
            return "" if Flaky.calls == 2 else "ok"

    cached = CachedLLM(Flaky(), Path(_TMP) / "flaky.sqlite")
    for expected in ("error", "", "ok"):
        try:
            assert cached.invoke("s", "u") == expected
        except LLMError:
            assert expected == "error"
    assert cached.entries() == 1


def test_demo_mode_never_needs_a_key():
    llm = get_llm(prefer="demo")
    assert llm.is_simulated and llm.notice


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


def test_extract_sql_variants():
    assert pl.extract_sql("Sure!\n```sql\nSELECT 1;\n```\nbye") == "SELECT 1;"
    assert pl.extract_sql("```\nSELECT 2\n```") == "SELECT 2"
    assert pl.extract_sql("Here: SELECT 3") == "SELECT 3"
    assert pl.extract_sql("WITH t AS (SELECT 1) SELECT * FROM t").startswith("WITH")


def test_attempt_classification():
    p = _pipeline()
    ok = p.run_attempt(pl.ZERO_SHOT_PROMPT, S1.question, S1.gold_sql)
    assert ok.succeeded and ok.status == "✅ correct" and pl.failure_category(ok) is None
    s3 = next(i for i in bm.BENCHMARK if i.id == "S3")
    err = p.run_attempt(pl.ZERO_SHOT_PROMPT, s3.question, s3.gold_sql)
    assert err.failure_type == "execution_error" and pl.failure_category(err) == "Unknown function"
    m4 = next(i for i in bm.BENCHMARK if i.id == "M4")
    wrong = p.run_attempt(pl.ZERO_SHOT_PROMPT, m4.question, m4.gold_sql)
    assert wrong.failure_type == "result_mismatch" and pl.failure_category(wrong) == "Wrong result"
    slow = p.run_attempt(pl.ZERO_SHOT_PROMPT, M20.question, M20.gold_sql)
    assert slow.failure_type == "inefficient_plan" and slow.efficiency_ratio > 3
    assert pl.failure_category(slow) == "Inefficient plan" and slow.plan.severe


def test_plan_awareness_can_be_switched_off():
    lenient = _pipeline(plan_aware=False).run_attempt(pl.ZERO_SHOT_PROMPT, M20.question, M20.gold_sql)
    assert lenient.correct and lenient.succeeded and not lenient.inefficient


def test_small_cost_differences_are_not_flagged_as_inefficient():
    join_sql = ("SELECT p.name FROM products p JOIN (SELECT category, AVG(price) a FROM products GROUP BY category) x "
                "ON x.category = p.category WHERE p.price > x.a")
    correlated = ("SELECT name FROM products p WHERE price > (SELECT AVG(price) FROM products WHERE category = p.category)")
    gold, naive = db.run_sql(join_sql), db.run_sql(correlated)
    assert naive.vm_steps / gold.vm_steps > 3, "precondition: the ratio alone would trigger the check"
    assert naive.vm_steps - gold.vm_steps < 3000, "precondition: but the absolute extra work is small"
    p = pl.AutoRefiningPipeline(_Scripted(f"```sql\n{correlated}\n```"))
    attempt = p.run_attempt("prompt", "products above their category average", join_sql)
    assert attempt.correct and attempt.plan.severe and not attempt.inefficient and attempt.succeeded


def test_refine_question_fixes_a_failing_query():
    p = _pipeline()
    s3 = next(i for i in bm.BENCHMARK if i.id == "S3")
    rounds = list(p.refine_question(s3.question, max_rounds=4))
    assert len(rounds) >= 2 and not rounds[0].attempt.succeeded and rounds[-1].attempt.succeeded
    assert rounds[0].gradient and rounds[0].refined_prompt and "strftime" in rounds[0].refined_prompt
    assert rounds[-1].gradient is None


def test_refine_question_on_unknown_question_in_demo_mode_raises_clear_error():
    try:
        list(_pipeline().refine_question("Which supplier ships the most parcels?"))
    except LLMError as exc:
        assert "built-in sample questions" in str(exc)
    else:
        raise AssertionError("expected LLMError")


def test_optimizer_only_trains_on_train_questions_and_never_gets_worse():
    p = _pipeline()
    seen: list[str] = []
    original = p.evaluate_prompt

    def spy(prompt, items, progress=None, label=""):
        seen.extend(i.id for i in items)
        return original(prompt, items, progress, label)

    p.evaluate_prompt = spy
    result = p.run_benchmark("multi", epochs=2, batch_size=3, size="Full")
    train_ids = {i.id for i in bm.items_for("multi", "train")}
    test_ids = {i.id for i in bm.items_for("multi", "test")}
    for step in result.optimization.steps:
        assert set(step.batch_ids) <= train_ids, "optimiser saw a test question"
    scores = [t[1] for t in result.optimization.trajectory]
    assert scores == sorted(scores), "accepted updates must never lower the train pass rate"
    assert test_ids <= set(seen), "test questions must be evaluated"
    assert result.n_train == 12 and result.n_test == 8


def test_benchmark_gains_in_every_tier_with_simulator():
    p = _pipeline()
    for tier in bm.TIER_LABELS:
        res = p.run_benchmark(tier, size="Full")
        assert res.textgrad.test.pass_rate > res.best_static.test.pass_rate, tier
        assert res.textgrad.test.exec_success >= res.zero_shot.test.exec_success


def test_multi_seed_run_keeps_every_run_and_selects_by_train_score():
    res = _pipeline().run_benchmark("multi", seeds=[0, 1, 2], size="Standard")
    assert len(res.runs) == 3
    best_train = max(r.optimization.final_train.score for r in res.runs)
    assert res.textgrad.train.score == best_train
    mean, sd, n = pl.seed_summary(res)
    assert n == 3 and 0 <= mean <= 100 and sd >= 0


def test_benchmark_refuses_custom_database():
    path = str(Path(_TMP) / "other.db")
    conn = sqlite3.connect(path); conn.execute("CREATE TABLE t(a)"); conn.commit(); conn.close()
    try:
        pl.AutoRefiningPipeline(MockLLM(), db_path=path).run_benchmark("single")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_estimate_calls_is_a_sane_upper_bound():
    llm = MockLLM()
    p = pl.AutoRefiningPipeline(llm)
    p.run_benchmark("advanced", epochs=2, batch_size=3, size="Full")
    assert llm.call_count <= pl.estimate_calls(12, 8, 2, 3, 1)


def test_all_scripted_flaws_really_fail():
    items = {i.id: i for i in bm.BENCHMARK}
    for item_id, chain in FAILURE_CHAINS.items():
        gold = db.run_sql(items[item_id].gold_sql)
        for rule, sql in chain:
            res = db.run_sql(sql)
            if res.ok:
                ok, _ = db.results_match(res, gold, db.has_order_by(items[item_id].gold_sql))
                slow = res.vm_steps / max(gold.vm_steps, 1) > 3 and res.vm_steps - gold.vm_steps >= 3000
                assert (not ok) or slow, f"{item_id}/{rule} is not actually a failure"


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #


def test_reports_render_and_flag_demo_mode():
    p = _pipeline()
    benchmarks = [p.run_benchmark(t, seeds=[0, 1], size="Quick") for t in bm.TIER_LABELS]
    md = report.build_markdown_report(benchmarks, "sim", True, {"size": "Quick"})
    assert "Demo mode" in md and "95% Wilson" in md and "| Tier |" in md and "Stability across seeds" in md
    import json

    data = json.loads(report.build_json_report(benchmarks, "sim", True))
    assert set(data["tiers"]) == {"Single-Table", "Multi-Join", "Advanced"} and data["simulated"] is True
    assert not pl.taxonomy_frame(benchmarks).empty and len(pl.paired_frame(benchmarks)) == 4


# --------------------------------------------------------------------------- #
# tiny runner so `python test_project.py` works without pytest
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
