"""
database.py
===========

A tiny, fully self-contained SQLite "sandbox" used by the whole project.

* Builds an e-commerce database with four related tables
  (customers, products, orders, order_items) from deterministic seed data,
  so every run (and every test) sees exactly the same rows.
* Exposes ``run_sql`` -- a safe, read-only executor that never raises: it
  returns a ``QueryResult`` holding either the rows or the exact database
  error message, which is the raw material for the TextGrad feedback loop.
* Exposes ``results_match`` -- an execution-accuracy comparison between a
  predicted result set and a gold result set.

Nothing here talks to an LLM, so this module can be tested in isolation:

    python database.py
"""

from __future__ import annotations

import itertools
import os
import random
import re
import sqlite3
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Location of the database file
# --------------------------------------------------------------------------- #

_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "ecommerce.db"

# Override with the TEXTGRAD_SQL_DB environment variable if you like.
DB_PATH = Path(os.environ.get("TEXTGRAD_SQL_DB", str(_DEFAULT_DB_PATH)))

# Path actually in use (may differ from DB_PATH if the folder is read-only and
# we had to fall back to the system temp directory).
_active_db_path: Path | None = None

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

# The same text is (a) executed to create the tables and (b) shown to the LLM
# as the schema, so the two can never drift apart.
SCHEMA_DDL = """\
-- customers: one row per registered customer
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    country     TEXT NOT NULL,   -- one of: 'USA', 'Canada', 'UK', 'Germany', 'India'
    signup_date TEXT NOT NULL    -- ISO date text, 'YYYY-MM-DD'
);

-- products: the catalogue
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,    -- one of: 'Electronics', 'Books', 'Home', 'Sports', 'Toys'
    price      REAL NOT NULL     -- current unit price in USD
);

-- orders: one row per order placed by a customer
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date  TEXT NOT NULL,   -- ISO date text, 'YYYY-MM-DD'
    status      TEXT NOT NULL    -- one of: 'completed', 'pending', 'cancelled', 'refunded'
);

-- order_items: the line items of each order (an order has 1-4 items)
CREATE TABLE order_items (
    item_id    INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity   INTEGER NOT NULL
);

-- Indexes. Note: order_items.product_id is NOT indexed, so looking products up in order_items
-- one by one (e.g. a correlated subquery per product) rescans the table; aggregate in one pass instead.
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_items_order     ON order_items(order_id);
"""

_TABLES = ("customers", "products", "orders", "order_items")
_N_INDEXES = 2


def get_schema_text(db_path: str | os.PathLike | None = None) -> str:
    """
    Schema shown to the LLM. For the built-in database this is the commented DDL;
    for any other SQLite file the schema is read from the file itself.
    """
    if db_path is None or Path(db_path).resolve() == get_db_path().resolve():
        return SCHEMA_DDL
    return introspect_schema(db_path)


def _ro_connect(path: str | os.PathLike) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def list_tables(db_path: str | os.PathLike) -> list[str]:
    """Names of all user tables in a SQLite file."""
    conn = _ro_connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def validate_sqlite_file(path: str | os.PathLike) -> tuple[bool, str]:
    """Check that ``path`` is a readable SQLite database with at least one table."""
    try:
        tables = list_tables(path)
    except sqlite3.Error as exc:
        return False, f"Not a readable SQLite database ({exc})."
    if not tables:
        return False, "The database contains no tables."
    return True, f"{len(tables)} table(s): {', '.join(tables[:12])}{' ...' if len(tables) > 12 else ''}"


def introspect_schema(db_path: str | os.PathLike, sample_rows: int = 2, max_tables: int = 40) -> str:
    """
    Describe an arbitrary SQLite database for the LLM: the CREATE statements stored in the
    file, its indexes, and for every table the row count plus a couple of example rows.
    """
    conn = _ro_connect(db_path)
    try:
        objects = conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY type DESC, name"
        ).fetchall()[:max_tables]
        parts: list[str] = []
        for kind, name, sql in objects:
            parts.append((sql or f"-- {kind} {name}").strip().rstrip(";") + ";")
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(name)}").fetchone()[0]
                cur = conn.execute(f"SELECT * FROM {_quote_ident(name)} LIMIT {int(sample_rows)}")
                cols = [d[0] for d in cur.description]
                parts.append(f"-- {name}: {count} rows")
                for row in cur.fetchall():
                    cells = ", ".join(f"{c}={str(v)[:30]!r}" for c, v in zip(cols, row))
                    parts.append(f"--   e.g. {cells}")
            except sqlite3.Error:
                parts.append(f"-- {name}: (could not be sampled)")
            parts.append("")
        indexes = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
        if indexes:
            parts.append("-- Indexes")
            parts += [r[0].strip().rstrip(";") + ";" for r in indexes]
        return "\n".join(parts).strip() + "\n"
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Deterministic seed data
# --------------------------------------------------------------------------- #

_CUSTOMER_NAMES = [
    "Alice Johnson", "Brian Smith", "Carla Gomez", "Daniel Kim", "Elena Petrova",
    "Farid Rahman", "Grace Liu", "Hassan Ali", "Isabel Costa", "Jonas Weber",
    "Katya Ivanova", "Liam O'Connor", "Maya Patel", "Noah Williams", "Olivia Brown",
    "Pedro Alvarez", "Quinn Taylor", "Rosa Martinez", "Samir Khan", "Tara Singh",
    "Umar Farouk", "Vera Novak", "Wei Zhang", "Ximena Torres", "Yusuf Demir",
    "Zoe Clarke", "Aaron Fischer", "Bianca Rossi", "Chen Wang", "Diana Lopez",
    "Ethan Moore", "Fatima Noor", "Gabriel Silva", "Hannah Muller", "Ivan Horvat",
    "Julia Schmidt", "Kofi Mensah", "Lena Hoffmann", "Mateo Ruiz", "Nina Kowalski",
]

_PRODUCTS = [  # (name, category, price)
    ("Wireless Mouse", "Electronics", 24.99),
    ("Mechanical Keyboard", "Electronics", 89.00),
    ("USB-C Hub", "Electronics", 39.50),
    ("Noise-Canceling Headphones", "Electronics", 199.99),
    ("4K Monitor", "Electronics", 329.00),
    ("Webcam", "Electronics", 59.95),
    ("Python Cookbook", "Books", 45.00),
    ("Data Science Handbook", "Books", 52.50),
    ("SQL in Practice", "Books", 38.00),
    ("Mystery Novel", "Books", 14.99),
    ("Cooking Basics", "Books", 22.50),
    ("Desk Lamp", "Home", 34.00),
    ("Coffee Maker", "Home", 79.99),
    ("Throw Blanket", "Home", 29.50),
    ("Air Purifier", "Home", 149.00),
    ("Yoga Mat", "Sports", 25.00),
    ("Dumbbell Set", "Sports", 119.00),
    ("Running Shoes", "Sports", 94.50),
    ("Building Blocks", "Toys", 31.00),
    ("Puzzle 1000pc", "Toys", 18.75),
    # Never ordered: gives "which product has no orders?" questions a non-empty answer.
    ("Sticker Pack", "Toys", 4.99),
]

# Order items are drawn only from the first 20 products, so the last one stays unsold
# (and the random seed data of the other tables is unchanged).
_N_SOLD_PRODUCTS = 20

# How many customers live in each country (sums to 40).
_COUNTRY_COUNTS = {"USA": 13, "Canada": 8, "UK": 8, "Germany": 6, "India": 5}

# How many customers per country never place an order. These make
# "customers who never ordered" / LEFT JOIN questions meaningful.
_NO_ORDER_QUOTA = {"USA": 2, "Germany": 2, "Canada": 1, "UK": 1, "India": 1}

_N_ORDERS = 120
_ORDER_STATUS_WEIGHTS = {"completed": 60, "pending": 15, "cancelled": 15, "refunded": 10}


def _build_rows(seed: int = 42) -> dict[str, list[tuple]]:
    """Generate all table rows deterministically."""
    rng = random.Random(seed)

    # --- customers ---------------------------------------------------------
    countries = [c for c, n in _COUNTRY_COUNTS.items() for _ in range(n)]
    rng.shuffle(countries)
    first_signup = date(2021, 1, 4)
    customers = []
    for i, (name, country) in enumerate(zip(_CUSTOMER_NAMES, countries), start=1):
        email = name.lower().replace(" ", ".").replace("'", "") + "@example.com"
        # Unique, strictly increasing signup dates (all before 2023).
        signup = first_signup + timedelta(days=17 * (i - 1))
        customers.append((i, name, email, country, signup.isoformat()))

    # --- which customers never order --------------------------------------
    quota = dict(_NO_ORDER_QUOTA)
    no_order: set[int] = set()
    for cust_id, _name, _email, country, _signup in reversed(customers):
        if quota.get(country, 0) > 0:
            no_order.add(cust_id)
            quota[country] -= 1
    eligible = [c[0] for c in customers if c[0] not in no_order]

    # --- products ----------------------------------------------------------
    products = [(i, n, cat, price) for i, (n, cat, price) in enumerate(_PRODUCTS, start=1)]

    # --- orders ------------------------------------------------------------
    # Every eligible customer gets at least one order, the rest are random.
    owners = list(eligible) + rng.choices(eligible, k=_N_ORDERS - len(eligible))
    rng.shuffle(owners)
    start = date(2023, 1, 1)
    offsets = sorted(rng.randrange(0, 730) for _ in range(_N_ORDERS))  # 2023-01-01 .. 2024-12-30
    statuses = rng.choices(
        list(_ORDER_STATUS_WEIGHTS), weights=list(_ORDER_STATUS_WEIGHTS.values()), k=_N_ORDERS
    )
    orders = [
        (i, owners[i - 1], (start + timedelta(days=offsets[i - 1])).isoformat(), statuses[i - 1])
        for i in range(1, _N_ORDERS + 1)
    ]

    # --- order items -------------------------------------------------------
    items: list[tuple] = []
    item_id = 1
    for order_id, *_ in orders:
        for product_id in rng.sample(range(1, _N_SOLD_PRODUCTS + 1), rng.randint(1, 4)):
            items.append((item_id, order_id, product_id, rng.randint(1, 5)))
            item_id += 1

    return {"customers": customers, "products": products, "orders": orders, "order_items": items}


def _create_database(path: Path) -> None:
    """Create ``path`` from scratch and fill it with the seed data."""
    if path.exists():
        path.unlink()
    rows = _build_rows()
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SCHEMA_DDL)
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", rows["customers"])
        conn.executemany("INSERT INTO products VALUES (?,?,?,?)", rows["products"])
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", rows["orders"])
        conn.executemany("INSERT INTO order_items VALUES (?,?,?,?)", rows["order_items"])
        conn.commit()
    finally:
        conn.close()


def _is_populated(path: Path) -> bool:
    """True if ``path`` already contains all four tables with data."""
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            filled = all(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] > 0 for t in _TABLES)
            # Also require the current catalogue and indexes, so an older database file is rebuilt automatically.
            n_indexes = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%'"
            ).fetchone()[0]
            return (
                filled
                and conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == len(_PRODUCTS)
                and n_indexes == _N_INDEXES
            )
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def init_database(db_path: str | os.PathLike | None = None, reset: bool = False) -> Path:
    """
    Make sure the sample database exists and return its path.

    Safe to call repeatedly. If the target folder is not writable the database
    is created in the system temp directory instead.
    """
    global _active_db_path
    path = Path(db_path) if db_path else DB_PATH
    if reset and path.exists():
        path.unlink()
    if not _is_populated(path):
        try:
            _create_database(path)
        except (OSError, sqlite3.OperationalError):
            path = Path(tempfile.gettempdir()) / "textgrad_sql_ecommerce.db"
            if reset or not _is_populated(path):
                _create_database(path)
    _active_db_path = path
    return path


def get_db_path() -> Path:
    """Path of the database currently in use (initialises it if needed)."""
    return _active_db_path or init_database()


def get_table_overview(db_path: str | os.PathLike | None = None) -> dict[str, int]:
    """Row count per table -- handy for the UI sidebar. Works for any SQLite file."""
    path = Path(db_path) if db_path else get_db_path()
    conn = _ro_connect(path)
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(t)}").fetchone()[0] for t in list_tables(path)}
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Safe query execution
# --------------------------------------------------------------------------- #


@dataclass
class QueryResult:
    """Outcome of executing one SQL string. Never raises; check ``ok``."""

    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False
    elapsed_ms: float = 0.0
    vm_steps: int = 0  # SQLite virtual-machine instructions executed: a deterministic cost measure

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_dataframe(self):
        """Result as a pandas DataFrame (imported lazily)."""
        import pandas as pd

        return pd.DataFrame(self.rows, columns=self.columns)

    def preview(self, n: int = 5) -> str:
        """Compact plain-text preview that is safe to paste into an LLM prompt."""
        if not self.ok:
            return f"<error: {self.error}>"
        if not self.rows:
            return f"columns: {', '.join(self.columns) or '-'}\n(0 rows)"
        lines = [f"columns: {', '.join(self.columns)}"]
        lines += [" | ".join(str(_norm_value(v)) for v in row) for row in self.rows[:n]]
        if len(self.rows) > n:
            lines.append(f"... ({len(self.rows)} rows in total)")
        return "\n".join(lines)


# Granularity of the VM-instruction counter (see run_sql).
VM_TICK = 25

# Only a single read-only statement is allowed: SELECT ... or WITH ... SELECT.
_READ_ONLY_RE = re.compile(r"^\s*(?:--[^\n]*\n\s*|/\*.*?\*/\s*)*(select|with)\b", re.IGNORECASE | re.DOTALL)


def run_sql(
    sql: str,
    db_path: str | os.PathLike | None = None,
    max_rows: int = 1000,
    timeout_s: float = 5.0,
) -> QueryResult:
    """
    Execute ``sql`` against the sample database and capture the outcome.

    * The database is opened read-only, so generated SQL can never modify it.
    * Anything that is not a single SELECT/WITH statement is rejected.
    * Runaway queries (e.g. accidental cross joins) are aborted after
      ``timeout_s`` seconds.
    * Every database exception is converted into ``QueryResult.error`` --
      that error text is exactly what the TextGrad critic gets to see.
    """
    sql = (sql or "").strip()
    if not sql:
        return QueryResult(ok=False, error="Empty SQL: the model returned no query.")
    if not _READ_ONLY_RE.match(sql):
        return QueryResult(ok=False, error="Rejected: only a single read-only SELECT / WITH query is allowed.")

    path = Path(db_path) if db_path else get_db_path()
    started = time.perf_counter()
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=timeout_s)
        conn.execute("PRAGMA query_only = ON")
        deadline = started + timeout_s
        ticks = [0]

        def _tick() -> int:
            # Called by SQLite every VM_TICK instructions: counts work and enforces the time limit.
            ticks[0] += 1
            return 1 if time.perf_counter() > deadline else 0

        conn.set_progress_handler(_tick, VM_TICK)
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        fetched = cursor.fetchmany(max_rows + 1)
        return QueryResult(
            ok=True,
            columns=columns,
            rows=[tuple(r) for r in fetched[:max_rows]],
            truncated=len(fetched) > max_rows,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            vm_steps=ticks[0] * VM_TICK,
        )
    except sqlite3.Error as exc:
        message = str(exc)
        if "interrupted" in message.lower():
            message = f"query exceeded the {timeout_s:.0f}s time limit and was aborted"
        return QueryResult(
            ok=False,
            error=f"{type(exc).__name__}: {message}",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:  # e.g. "You can only execute one statement at a time"
        return QueryResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
    finally:
        if conn is not None:
            conn.close()


# --------------------------------------------------------------------------- #
# Execution-accuracy comparison
# --------------------------------------------------------------------------- #


def _norm_value(value: Any) -> Any:
    """Normalise a cell so 24.990000000001 and 24.99 compare equal."""
    if isinstance(value, float):
        return round(value, 2)
    return value


def has_order_by(sql: str) -> bool:
    """True if the (gold) query defines an explicit row order."""
    return re.search(r"\border\s+by\b", sql or "", re.IGNORECASE) is not None


def results_match(pred: QueryResult, gold: QueryResult, ordered: bool = False) -> tuple[bool, str]:
    """
    Compare a predicted result with the gold result ("execution accuracy").

    * Column names are ignored and columns may appear in a different order.
    * Rows are compared as a multiset unless ``ordered`` is True.
    * Floats are compared to 2 decimals.

    Returns ``(matches, reason)``; ``reason`` explains a mismatch in one line.
    """
    if not pred.ok:
        return False, f"the query failed to execute ({pred.error})"
    if not gold.ok:
        return False, "the reference query failed to execute"
    if pred.row_count != gold.row_count:
        return False, f"it returned {pred.row_count} row(s) but {gold.row_count} were expected"

    n_pred = len(pred.columns)
    n_gold = len(gold.columns)
    if n_pred != n_gold:
        return False, f"it returned {n_pred} column(s) but {n_gold} were expected"

    gold_rows = [tuple(_norm_value(v) for v in row) for row in gold.rows]
    gold_key = gold_rows if ordered else Counter(gold_rows)

    # Allow a different column order (cheap: at most 5! = 120 permutations).
    permutations = itertools.permutations(range(n_pred)) if n_pred <= 5 else [tuple(range(n_pred))]
    for perm in permutations:
        pred_rows = [tuple(_norm_value(row[i]) for i in perm) for row in pred.rows]
        if (pred_rows if ordered else Counter(pred_rows)) == gold_key:
            return True, "result matches the expected result"
    return False, "the number of rows is right but the values differ from the expected result"


if __name__ == "__main__":
    path = init_database(reset=True)
    print(f"Database created at {path}")
    for table, count in get_table_overview(path).items():
        print(f"  {table:12s} {count:4d} rows")
    demo = run_sql("SELECT country, COUNT(*) AS n FROM customers GROUP BY country ORDER BY n DESC")
    print("\nCustomers per country:\n" + demo.preview(10))
    broken = run_sql("SELECT YEAR(order_date) FROM orders")
    print("\nCaptured error example:\n  " + str(broken.error))
