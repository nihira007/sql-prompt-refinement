# Iterative Prompt Refinement for SQL Query Planning via TextGrad

An auto-refining, **plan-aware** text-to-SQL engine. A language model writes SQL, SQLite executes it, and when
the query fails, returns the wrong result, or returns the right result through a **slow query plan**, a
**TextGrad-style** loop turns that failure into textual feedback and **rewrites the system prompt** so later
queries are better. A Streamlit dashboard shows every step live and compares the optimised prompt with static
zero-shot and few-shot prompting on **three complexity tiers**, with confidence intervals and significance tests.
It also works on **your own SQLite database**.

```
question + schema ──(LLM + system prompt)──► SQL ──(SQLite)──► rows │ error
                              ▲                        │
                              │                        └─► EXPLAIN QUERY PLAN + measured cost
                              │                                         │
                              └── optimiser rewrites the prompt ◄── critic LLM writes a textual gradient
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10+. With no model configured the app starts in **demo mode** (see below). To use a real model, set the
environment variables below (or copy `.env.example` to `.env`) **before** starting the app.

### Free and local: Ollama (no API key)

```bash
ollama pull qwen2.5-coder:7b            # one-time download, then make sure Ollama is running
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama            # any non-empty text; Ollama ignores it
export OPENAI_MODEL=qwen2.5-coder:7b
streamlit run app.py
```

Windows PowerShell: `$env:OPENAI_BASE_URL="http://localhost:11434/v1"` and so on. Other OpenAI-compatible
providers (OpenAI, Groq, Gemini, vLLM, a LiteLLM proxy, ...) work the same way with their base URL, key and model.
Restart the app after changing variables. Local models on CPU are slow: keep **Benchmark size = Quick** (the
default), and let the response cache carry long runs (see below).

| Variable             | Meaning                                                                                  |
|----------------------|------------------------------------------------------------------------------------------|
| `OPENAI_API_KEY`     | Key for the endpoint. If missing, the app runs in demo mode.                             |
| `OPENAI_MODEL`       | Model name (default `gpt-4o-mini`).                                                      |
| `OPENAI_BASE_URL`    | Optional. Any OpenAI-compatible endpoint.                                                |
| `OPENAI_TIMEOUT`     | Seconds per request (default 180; raise it for slow local models).                       |
| `TEXTGRAD_SQL_CACHE` | `0` disables the on-disk response cache; a path chooses where it lives. Default: on.     |
| `TEXTGRAD_SQL_DB`    | Where the built-in SQLite file is created (default `./ecommerce.db`).                    |

## What is in the dashboard

**🔍 Live refinement.** Pick a sample question (or write your own), press *Generate & refine*, and watch each
round: the initial SQL, the execution result or the exact database error, the **query plan and its measured cost**,
the TextGrad feedback, and the refined prompt with a diff. Choose zero-shot, few-shot or the prompt TextGrad
learned in the benchmark as the starting point. A reference SQL is optional and enables correctness and
efficiency checks on any question.

**📊 Benchmark.** Three strategies on three tiers, scored on **held-out test questions**:

1. static zero-shot prompt
2. static few-shot prompt
3. the zero-shot prompt optimised by TextGrad on the **train** questions only

You get headline numbers, a summary table with **95% confidence intervals**, a bar chart with CI whiskers, a
**paired significance test** against the stronger static baseline, a **failure taxonomy** (what kind of mistakes
each strategy makes), training curves, per-question outcomes, every optimiser step with its feedback and prompt
diff, and **downloads** (Markdown report, JSON results, learned prompts).

**ℹ️ How it works.** The loop, the metrics and the design choices on one page.

**Your own database.** Choose *Upload my own SQLite file* in the sidebar. The schema (including example rows and
indexes) is read automatically and you can refine questions about your own data. The benchmark itself needs the
built-in database because it relies on labelled reference queries.

## The query-planning part

Correct SQL can still be bad SQL. For every query that runs, the pipeline

* reads SQLite's `EXPLAIN QUERY PLAN` and flags anti-patterns such as correlated subqueries that re-scan a table for
  every row, multiple full scans, and temporary sort structures;
* measures the real work in **SQLite VM steps** (a deterministic cost measure, unlike wall-clock time);
* compares it with the reference query. A correct result that needs **more than 3x the work and at least 3000
  extra steps** counts as a failure (the absolute floor stops tiny queries from being flagged by noise).

The plan, the cost ratio and the detected problems become part of the *loss* that is back-propagated to the prompt.
The sample schema deliberately leaves `order_items.product_id` unindexed so such questions have a measurable
slow variant. Turn the check off in the sidebar to score only correctness.

## Metrics

| Metric             | Meaning                                                                                    |
|--------------------|--------------------------------------------------------------------------------------------|
| Execution success  | the query ran without a database error                                                     |
| Execution accuracy | the result equals the reference result (column order ignored, rows compared as a multiset) |
| **Pass rate**      | correct **and** not an inefficient plan; the headline metric                               |
| 95% CI             | Wilson score interval over the test questions                                              |
| p-value            | exact two-sided sign test, TextGrad vs the stronger static prompt, same questions          |

## The benchmark

60 labelled questions with reference SQL over a four-table e-commerce database (`customers`, `products`,
`orders`, `order_items`), 12 train and 8 test per tier:

| Tier         | Covers                                                                                  |
|--------------|-----------------------------------------------------------------------------------------|
| Single-Table | filters, aggregates, GROUP BY / HAVING, ORDER BY / LIMIT, date functions                |
| Multi-Join   | 2-4 joined tables, INNER / LEFT JOIN, DISTINCT counts, revenue maths, zero-filled joins |
| Advanced     | subqueries, CTEs, window functions, conditional aggregation, "for every ..." questions  |

Questions have no ties (so there is one right answer), and `python benchmark.py` verifies that every reference
query runs, returns rows and fits its tier.

Sizes: **Quick** (4 train / 4 test per tier, the default), **Standard** (8 / 6) and **Full** (12 / 8). Optional
**multiple optimisation runs (seeds)** show how stable the result is; the headline run is chosen by its *train*
score, never by its test score.

## Demo mode (no model configured)

The app does not crash without a model. It uses an **offline simulator** (`mock_llm.py`) that is *scripted*, not a
language model: it makes typical text-to-SQL mistakes (T-SQL `TOP`, `YEAR()`, aggregates in `WHERE`, missing
`LEFT JOIN`, nested aggregates, correlated subqueries, ...) unless the prompt contains the guideline that fixes the
mistake, and it only answers the built-in questions. It lets you see all the machinery work offline.

**Demo-mode numbers demonstrate the pipeline, not model quality**, and the app and reports say so. Example
(simulator, Full size, one run, pass rate on the 8 held-out test questions per tier):

| Tier         | Zero-shot | Few-shot | TextGrad |
|--------------|----------:|---------:|---------:|
| Single-Table |       25% |      50% |     100% |
| Multi-Join   |       25% |      25% |      88% |
| Advanced     |        0% |       0% |      75% |

TextGrad does not reach 100% on every tier, and that is intended: some test questions need a guideline that the
training questions never exposed (for example date functions or the "completed orders only" rule in the Advanced
tier), or one that a given run's training batches reached too late. That is the generalisation gap a held-out
test exists to expose. The simulator is deterministic, so these numbers repeat exactly; with a real model the
gains will differ and may be smaller.

## Design notes

* **Validation-gated updates.** A candidate prompt is kept only if the train score strictly improves, so a bad
  rewrite cannot make the prompt worse.
* **Held-out evaluation.** The optimiser learns from train questions only; the critic is told to give general
  feedback, not question-specific answers.
* **Safe execution.** Read-only connection, single `SELECT`/`WITH` only, runaway queries aborted after 5 seconds.
* **Protected output format.** A fixed footer ("return one SQLite SELECT in a ```sql block") is always appended so
  the optimiser cannot rewrite it away.
* **Resumable runs.** With a live model, responses are cached in `.llm_cache.sqlite` (temperature 0). An interrupted
  benchmark continues where it stopped, repeats are free, and the sidebar shows the call estimate up front. Use
  *Clear response cache* after changing prompts' wording in code or switching models of the same name.

## Project structure

```
app.py                 Streamlit dashboard (3 tabs, upload your own DB)
pipeline.py            generate -> execute -> plan -> evaluate -> TextGrad loop, training, benchmark, tables
textgrad_optimizer.py  textual loss, gradients (critic) and the prompt update step (optimiser)
query_plan.py          EXPLAIN QUERY PLAN parsing and anti-pattern detection
database.py            sample database, safe run_sql with VM-step cost, result comparison, schema introspection
benchmark.py           60 labelled questions, 3 tiers, train/test split, self-validation
stats.py               Wilson intervals, sign test, paired counts
report.py              Markdown / JSON report export
llm_client.py          LangChain (ChatOpenAI) client, OpenAI-SDK fallback, disk cache, demo fallback
mock_llm.py            offline simulator used in demo mode
test_project.py        34 tests (no network, no Streamlit needed)
requirements.txt       dependencies
.env.example           template for the model settings
```

## Useful commands

```bash
python test_project.py     # or: pytest -q test_project.py
python benchmark.py        # verify the 60 reference queries
python database.py         # rebuild the sample database
```

## Limitations

* A tier's test set has 8 questions, so one question is 12.5 percentage points. The confidence intervals and
  p-values are there to show how much each tier can and cannot prove; use **Full** size and several seeds for
  tighter evidence.
* Execution accuracy compares result sets, so two different but equivalent queries both pass; questions whose
  reference has `ORDER BY` also compare row order.
* The efficiency check is relative to the reference query and tuned for this small database (thresholds in
  `AutoRefiningPipeline`). On your own database, without a reference SQL, only correlated subqueries are flagged.
* Small local models are weaker at SQL and at rewriting prompts; the validation gate rejects bad rewrites, so the
  worst case is "no improvement", not a worse prompt.
