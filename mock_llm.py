"""
mock_llm.py
===========

Offline *simulator* used when no API key / local model is configured, so the whole
project (pipeline, TextGrad loop, benchmark, Streamlit UI) runs out of the box.

IMPORTANT -- what this is and is not
------------------------------------
This is NOT a language model. It is a scripted stand-in that behaves like a
prompt-sensitive model on the built-in benchmark questions:

* generate : returns a flawed SQL query unless the system prompt already contains the
             guideline that fixes that particular flaw, otherwise the correct query.
* gradient : writes critic feedback that names the missing guideline.
* optimize : appends the guidelines mentioned in the feedback to the prompt
             (at most ``max_edits_per_step`` per step, like a small learning rate).

Because the behaviour is scripted, accuracy numbers produced in demo mode demonstrate
that the *machinery* works (errors and slow plans are captured, feedback is propagated,
prompts are rewritten, edits are accepted/rejected on measured accuracy). They are NOT
evidence about how well a real LLM performs. Use a real model to measure that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from benchmark import find_item_fuzzy
from llm_client import BaseLLM, LLMError


@dataclass(frozen=True)
class Rule:
    """One prompt guideline the simulator knows about."""

    id: str
    marker: re.Pattern  # "does this text contain the guideline?"
    instruction: str  # the line the optimizer adds to the prompt
    advice: str  # what the critic says about it


RULES: dict[str, Rule] = {
    r.id: r
    for r in [
        Rule(
            "limit",
            re.compile(r"\bLIMIT\b"),
            "- SQLite has no TOP keyword. For 'top N' or 'N most/least ...' questions use ORDER BY <column> "
            "DESC (or ASC) followed by LIMIT N.",
            "The query used T-SQL style TOP, which SQLite rejects. The prompt should state that SQLite has no "
            "TOP keyword and that top-N questions need ORDER BY followed by LIMIT N.",
        ),
        Rule(
            "dates",
            re.compile(r"strftime"),
            "- SQLite has no YEAR(), MONTH() or DATE_TRUNC(). Dates are ISO text ('YYYY-MM-DD'); filter and group "
            "them with strftime, e.g. strftime('%Y', order_date) = '2024' or strftime('%Y-%m', order_date).",
            "The query called YEAR()/MONTH(), which do not exist in SQLite. The prompt should say that dates are "
            "ISO text and must be filtered and grouped with strftime(...).",
        ),
        Rule(
            "schema_columns",
            re.compile(r"quantity\s*\*\s*(\w+\.)?price"),
            "- Use only columns that exist in the schema; keys are named <table>_id (never just id). No table "
            "stores a total amount: compute revenue as SUM(order_items.quantity * products.price).",
            "The query referenced a column that does not exist. The prompt should tell the model to use only "
            "schema columns and that revenue has to be computed as SUM(order_items.quantity * products.price).",
        ),
        Rule(
            "completed_only",
            re.compile(r"status\s*=\s*'completed'"),
            "- Revenue and sales figures count only orders with status = 'completed'; exclude pending, "
            "cancelled and refunded orders.",
            "The query ran but included orders that should not count. The prompt should say that revenue and "
            "sales are computed only from orders with status = 'completed'.",
        ),
        Rule(
            "having",
            re.compile(r"never in WHERE"),
            "- Filter on aggregate values (COUNT, SUM, AVG) with HAVING after GROUP BY, never in WHERE.",
            "The query put an aggregate condition in WHERE. The prompt should require HAVING after GROUP BY "
            "for conditions on COUNT/SUM/AVG, never in WHERE.",
        ),
        Rule(
            "distinct_count",
            re.compile(r"COUNT\(DISTINCT <id>\)"),
            "- When counting entities (customers, products, orders) across joins, use COUNT(DISTINCT <id>) so "
            "rows duplicated by the join are not counted twice.",
            "The count was inflated by rows duplicated through the join. The prompt should require "
            "COUNT(DISTINCT <id>) when counting entities across joined tables.",
        ),
        Rule(
            "dedupe",
            re.compile(r"SELECT DISTINCT"),
            "- When listing entities (customers, products) found through joins, use SELECT DISTINCT so an entity "
            "matched several times appears only once.",
            "The list contains the same entity several times because of the joins. The prompt should say to use "
            "SELECT DISTINCT when listing entities found through joins.",
        ),
        Rule(
            "left_join",
            re.compile(r"\bLEFT JOIN\b"),
            "- For 'never', 'no', 'without' or 'including those with none' questions use LEFT JOIN and test the "
            "right side for NULL (or count it) instead of an INNER JOIN.",
            "An INNER JOIN dropped the rows that have no match. The prompt should say to use LEFT JOIN for "
            "'never / without / including those with none' questions.",
        ),
        Rule(
            "coalesce",
            re.compile(r"COALESCE"),
            "- When entries without matches must show 0, use an outer join and wrap the aggregate: "
            "COALESCE(SUM(col), 0).",
            "Entries without matches show NULL instead of 0. The prompt should say to wrap aggregates in "
            "COALESCE(SUM(col), 0) when unmatched entries must show 0.",
        ),
        Rule(
            "qualify",
            re.compile(r"\b[a-z]{1,2}\.\w+\s*=\s*[a-z]{1,2}\.\w+"),
            "- When joining tables, give every table a short alias and qualify every column with it, e.g. "
            "JOIN orders o ON c.customer_id = o.customer_id, to avoid ambiguous column names.",
            "The query joined tables without aliases, so columns such as customer_id are ambiguous. The prompt "
            "should require short table aliases and qualified columns, e.g. JOIN orders o ON c.customer_id = "
            "o.customer_id.",
        ),
        Rule(
            "cte",
            re.compile(r"\bWITH\b"),
            "- Aggregates cannot be nested (AVG(SUM(x)) is invalid). Compute the inner aggregate first in a CTE "
            "(WITH totals AS (SELECT ...)) or a derived table, then compare against (SELECT AVG(col) FROM totals).",
            "The query nested aggregates such as AVG(SUM(...)), which SQLite rejects. The prompt should say to "
            "compute the inner aggregate first in a CTE (WITH ... AS (...)) or derived table and compare "
            "against a subquery on it.",
        ),
        Rule(
            "window",
            re.compile(r"\bOVER\b"),
            "- SQLite supports window functions. For 'rank', 'top N per group' and 'first per group' questions use "
            "ROW_NUMBER() / RANK() OVER (PARTITION BY <group> ORDER BY <column>) in a subquery and filter on it.",
            "The query could not express a ranking or per-group selection. The prompt should say to use window "
            "functions (ROW_NUMBER() / RANK() OVER (PARTITION BY ... ORDER BY ...)) for rank and top-N-per-group "
            "questions.",
        ),
        Rule(
            "case_when",
            re.compile(r"CASE WHEN"),
            "- Use SUM(CASE WHEN <condition> THEN 1 ELSE 0 END) for conditional counts and for 'all / every' tests "
            "(compare it with COUNT(*)). For percentages multiply by 100.0, not 100, to avoid integer division.",
            "The conditional logic was wrong (an 'all' test written as 'any', or integer division in a "
            "percentage). The prompt should say to use SUM(CASE WHEN ... THEN 1 ELSE 0 END) and to multiply by "
            "100.0 for percentages.",
        ),
        Rule(
            "division",
            re.compile(r"'every'"),
            "- For 'every' / 'all' questions (customers who ordered from every category) compare the distinct "
            "matches with the total number of groups: HAVING COUNT(DISTINCT p.category) = (SELECT "
            "COUNT(DISTINCT category) FROM products).",
            "The 'ordered from every category' question needs a division-style comparison. The prompt should say "
            "that for 'every' questions the distinct matches are compared with the total number of groups in a "
            "subquery.",
        ),
        Rule(
            "set_based",
            re.compile(r"correlated"),
            "- Avoid correlated subqueries that re-scan a table once per outer row (for example one scalar "
            "subquery per product); order_items.product_id has no index. Aggregate once with a JOIN and GROUP BY "
            "(or a CTE) instead.",
            "The query is correct but inefficient: a correlated subquery re-scans a table for every outer row. "
            "The prompt should say to avoid correlated subqueries and to aggregate once with a JOIN and GROUP BY.",
        ),
    ]
}

# --------------------------------------------------------------------------- #
# Flawed queries: for each benchmark question, the ordered chain of mistakes a
# model *without* the matching guideline makes. The first rule missing from the
# prompt decides which flawed query is returned; if none is missing, the gold
# query is used. Questions without a chain are "easy": every prompt solves them.
# --------------------------------------------------------------------------- #

_OL = "FROM orders o JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id"
_C4 = (
    "FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
    "JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id"
)

FAILURE_CHAINS: dict[str, list[tuple[str, str]]] = {
    # ---- single table -------------------------------------------------------
    "S2": [("limit", "SELECT TOP 3 name FROM products ORDER BY price DESC;")],
    "S3": [("dates", "SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = 2024;")],
    "S4": [("having", "SELECT category, COUNT(*) AS product_count FROM products WHERE COUNT(*) > 3 GROUP BY category;")],
    "S7": [("dates", "SELECT COUNT(*) FROM customers WHERE YEAR(signup_date) = 2022;")],
    "S11": [("having", "SELECT order_id, COUNT(DISTINCT product_id) FROM order_items WHERE COUNT(DISTINCT product_id) > 2 GROUP BY order_id;")],
    "S13": [("dates", "SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = 2024 AND MONTH(order_date) = 3;")],
    "S14": [("having", "SELECT country, COUNT(*) AS customer_count FROM customers WHERE COUNT(*) >= 7 GROUP BY country;")],
    "S15": [("limit", "SELECT TOP 3 name, signup_date FROM customers ORDER BY signup_date DESC;")],
    "S17": [("limit", "SELECT TOP 5 name, price FROM products ORDER BY price ASC;")],
    "S19": [("having", "SELECT product_id, COUNT(*) FROM order_items WHERE COUNT(*) > 10 GROUP BY product_id;")],
    "S20": [("dates", "SELECT YEAR(order_date) AS year, COUNT(*) FROM orders GROUP BY YEAR(order_date);")],
    # ---- multi join ---------------------------------------------------------
    "M1": [("qualify", "SELECT name, COUNT(order_id) AS order_count FROM customers JOIN orders ON customer_id = customer_id GROUP BY name;")],
    "M2": [
        ("schema_columns", "SELECT SUM(total_amount) AS total_revenue FROM orders;"),
        ("completed_only", "SELECT SUM(oi.quantity * p.price) AS total_revenue FROM order_items oi JOIN products p ON oi.product_id = p.product_id;"),
    ],
    "M3": [("distinct_count", "SELECT COUNT(c.customer_id) FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE c.country = 'USA';")],
    "M4": [("left_join", "SELECT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_id IS NULL;")],
    "M5": [("having", "SELECT c.name, COUNT(o.order_id) AS order_count FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE COUNT(o.order_id) > 3 GROUP BY c.customer_id, c.name;")],
    "M6": [("dates", "SELECT COUNT(*) FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE c.country = 'Canada' AND YEAR(o.order_date) = 2024;")],
    "M9": [("dedupe", f"SELECT c.name {_C4} WHERE p.category = 'Toys';")],
    "M11": [
        ("left_join", "SELECT p.name, SUM(oi.quantity) AS total_quantity FROM products p JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.product_id, p.name;"),
        ("coalesce", "SELECT p.name, SUM(oi.quantity) AS total_quantity FROM products p LEFT JOIN order_items oi ON oi.product_id = p.product_id GROUP BY p.product_id, p.name;"),
        ("set_based", "SELECT p.name, COALESCE((SELECT SUM(oi.quantity) FROM order_items oi WHERE oi.product_id = p.product_id), 0) AS total_quantity FROM products p;"),
    ],
    "M12": [("dedupe", f"SELECT c.name {_C4} WHERE p.name = 'Wireless Mouse';")],
    "M13": [
        ("schema_columns", "SELECT p.category, SUM(oi.total_price) AS revenue FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.category ORDER BY revenue DESC LIMIT 3;"),
        ("completed_only", "SELECT p.category, SUM(oi.quantity * p.price) AS revenue FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.category ORDER BY revenue DESC LIMIT 3;"),
    ],
    "M14": [("left_join", "SELECT c.name, COUNT(o.order_id) AS order_count FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE c.country = 'Germany' GROUP BY c.customer_id, c.name;")],
    "M15": [
        ("qualify", "SELECT category, SUM(quantity) AS total_quantity FROM orders JOIN order_items ON order_id = order_id JOIN products ON product_id = product_id WHERE status = 'completed' AND strftime('%Y', order_date) = '2024' GROUP BY category;"),
        ("dates", f"SELECT p.category, SUM(oi.quantity) AS total_quantity {_OL} WHERE o.status = 'completed' AND YEAR(o.order_date) = 2024 GROUP BY p.category;"),
    ],
    "M16": [("distinct_count", "SELECT COUNT(oi.product_id) FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id WHERE c.country = 'UK';")],
    "M19": [("distinct_count", f"SELECT p.category, COUNT(o.customer_id) AS customers {_OL} GROUP BY p.category;")],
    "M20": [
        ("set_based","SELECT p.name, (SELECT COUNT(DISTINCT oi.order_id) FROM order_items oi WHERE oi.product_id = p.product_id) AS orders_containing FROM products p;"),
    ],
    # ---- advanced -----------------------------------------------------------
    "A1": [("cte", f"SELECT c.name, SUM(oi.quantity * p.price) AS total {_C4} WHERE o.status = 'completed' GROUP BY c.customer_id, c.name HAVING SUM(oi.quantity * p.price) > AVG(SUM(oi.quantity * p.price));")],
    "A2": [("left_join", "SELECT p.name FROM products p JOIN order_items oi ON oi.product_id = p.product_id WHERE oi.product_id IS NULL;")],
    "A3": [("case_when", "SELECT ROUND(100 * SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*), 1) FROM orders;")],
    "A4": [("case_when", "SELECT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.status = 'completed' GROUP BY c.customer_id, c.name;")],
    "A5": [("cte", f"SELECT AVG(SUM(oi.quantity * p.price)) {_OL} WHERE o.status = 'completed' GROUP BY o.order_id;")],
    "A6": [("window", f"SELECT p.category, SUM(oi.quantity) AS qty, RANK() AS rnk {_OL} WHERE o.status = 'completed' GROUP BY p.category;")],
    "A9": [("case_when", "SELECT ROUND(100 * COUNT(DISTINCT o.customer_id) / (SELECT COUNT(*) FROM customers), 1) FROM orders o;")],
    "A10": [("division", f"SELECT c.name {_C4} GROUP BY c.customer_id, c.name HAVING COUNT(DISTINCT p.category) = (SELECT COUNT(*) FROM categories);")],
    "A11": [("window", f"SELECT c.country, c.name, SUM(oi.quantity * p.price) AS total {_C4} WHERE o.status = 'completed' GROUP BY c.customer_id, c.name, c.country ORDER BY total DESC LIMIT 2;")],
    "A13": [("cte", "SELECT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.name HAVING COUNT(o.order_id) > AVG(COUNT(o.order_id));")],
    "A14": [("window", "SELECT category, name, price FROM products ORDER BY price DESC LIMIT 5;")],
    "A15": [("dates", "SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS n FROM orders WHERE YEAR(order_date) = 2024 GROUP BY month ORDER BY n DESC LIMIT 1;")],
    "A16": [("left_join", "SELECT COUNT(DISTINCT c.customer_id) FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.status <> 'completed';")],
    "A17": [("completed_only", f"SELECT ROUND(100.0 * SUM(CASE WHEN p.category = 'Electronics' THEN oi.quantity * p.price ELSE 0 END) / SUM(oi.quantity * p.price), 1) {_OL};")],
    "A18": [("cte", f"SELECT o.order_id, SUM(oi.quantity * p.price) AS total {_OL} GROUP BY o.order_id HAVING SUM(oi.quantity * p.price) > AVG(SUM(oi.quantity * p.price));")],
    "A19": [("cte", f"SELECT c.name {_C4.replace('JOIN products p ON oi.product_id = p.product_id', '')} GROUP BY c.customer_id, c.name HAVING COUNT(DISTINCT oi.product_id) > AVG(COUNT(DISTINCT oi.product_id));")],
    "A20": [("window", "SELECT country, name, MIN(signup_date) FROM customers GROUP BY country;")],
}

LEARNED_HEADER = "Learned guidelines:"


def _tag(text: str, name: str) -> str:
    """Extract the content of <NAME>...</NAME> from ``text`` ('' if absent)."""
    match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def first_missing_rule(item_id: str, prompt: str) -> tuple[str, str] | tuple[None, None]:
    """First rule of the item's failure chain that the prompt does not contain yet."""
    for rule_id, flawed_sql in FAILURE_CHAINS.get(item_id, []):
        if not RULES[rule_id].marker.search(prompt):
            return rule_id, flawed_sql
    return None, None


class MockLLM(BaseLLM):
    """Scripted stand-in for the LLM (see module docstring)."""

    name = "Offline simulator (demo mode)"
    is_simulated = True

    def __init__(self, max_edits_per_step: int = 2) -> None:
        super().__init__()
        self.max_edits_per_step = max_edits_per_step

    # ------------------------------------------------------------------ #
    def _invoke(self, system: str, user: str, role: str) -> str:
        if role == "generate":
            return self._generate(system, user)
        if role == "gradient":
            return self._gradient(user)
        if role == "optimize":
            return self._optimize(user)
        raise LLMError(f"MockLLM does not know the role {role!r}.")

    # ------------------------------------------------------------------ #
    def _generate(self, system: str, user: str) -> str:
        question = _tag(user, "QUESTION")
        item = find_item_fuzzy(question)
        if item is None:
            raise LLMError(
                "Demo mode can only answer the built-in sample questions. Pick one from the list, "
                "or configure a real model (see README) to ask free-form questions."
            )
        rule_id, flawed_sql = first_missing_rule(item.id, system)
        sql = flawed_sql if rule_id else item.gold_sql
        return f"```sql\n{sql}\n```"

    def _gradient(self, user: str) -> str:
        prompt = _tag(user, "PROMPT")
        item = find_item_fuzzy(_tag(user, "QUESTION"))
        rule_id = first_missing_rule(item.id, prompt)[0] if item else None
        if rule_id is None:
            return (
                "The failure does not match any guideline this simulator knows about. The prompt should ask the "
                "model to re-check table names, join keys and filters against the schema before answering."
            )
        return f"Diagnosis: {RULES[rule_id].advice}"

    def _optimize(self, user: str) -> str:
        current = _tag(user, "CURRENT_PROMPT")
        feedback = _tag(user, "FEEDBACK")
        new_rules = [
            r for r in RULES.values() if r.marker.search(feedback) and not r.marker.search(current)
        ][: self.max_edits_per_step]
        if not new_rules:
            improved = current
        else:
            lines = "\n".join(r.instruction for r in new_rules)
            if LEARNED_HEADER in current:
                improved = f"{current.rstrip()}\n{lines}"
            else:
                improved = f"{current.rstrip()}\n\n{LEARNED_HEADER}\n{lines}"
        return f"<IMPROVED_PROMPT>\n{improved}\n</IMPROVED_PROMPT>"
