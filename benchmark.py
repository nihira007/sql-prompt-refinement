"""
benchmark.py
============

A labelled text-to-SQL benchmark over the e-commerce database: 60 questions.

Every item has a natural-language question and a *gold* SQL query. The gold query's
result set is what a predicted query is compared against ("execution accuracy"), and
its measured cost (SQLite VM steps) is the yardstick for query-plan efficiency.

Items are organised along two axes:

* tier  -- "single"   one table
           "multi"    two or more joined tables
           "advanced" subqueries, CTEs, window functions, conditional aggregation
* split -- "train" (the optimizer may learn from these failures) or
           "test"  (held out: only used to measure how well the learned prompt generalises)

Each tier has 12 train and 8 test questions. Run ``python benchmark.py`` to check that
every gold query executes, returns rows, and fits its tier.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

TIER_LABELS = {"single": "Single-Table", "multi": "Multi-Join", "advanced": "Advanced"}
TIER_DESCRIPTIONS = {
    "single": "one table: filters, aggregates, GROUP BY / HAVING, ORDER BY / LIMIT, date functions",
    "multi": "2-4 joined tables: INNER / LEFT JOIN, DISTINCT counts, revenue maths",
    "advanced": "subqueries, CTEs, window functions, conditional aggregation, division queries",
}
LABEL_TO_TIER = {v: k for k, v in TIER_LABELS.items()}

# (train, test) questions per tier used by the three benchmark sizes
SIZES = {"Quick": (4, 4), "Standard": (8, 6), "Full": (12, 8)}


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    tier: str  # "single" | "multi" | "advanced"
    split: str  # "train" | "test"
    question: str
    gold_sql: str


def _item(id_: str, tier: str, split: str, question: str, gold_sql: str) -> BenchmarkItem:
    return BenchmarkItem(id_, tier, split, question, " ".join(gold_sql.split()))


# Reused pieces of gold SQL
_ORDER_LINES = (
    "FROM orders o JOIN order_items oi ON o.order_id = oi.order_id "
    "JOIN products p ON oi.product_id = p.product_id"
)
_SPEND_CTE = (
    "spend AS (SELECT c.customer_id, c.name, c.country, SUM(oi.quantity * p.price) AS total "
    "FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
    "JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id "
    "WHERE o.status = 'completed' GROUP BY c.customer_id, c.name, c.country)"
)

BENCHMARK: list[BenchmarkItem] = [
    # ================================================================== #
    # Tier 1: single table  (S1-S12 train, S13-S20 test)
    # ================================================================== #
    _item("S1", "single", "train", "How many customers are from Canada?",
          "SELECT COUNT(*) FROM customers WHERE country = 'Canada';"),
    _item("S2", "single", "train", "List the names of the 3 most expensive products.",
          "SELECT name FROM products ORDER BY price DESC LIMIT 3;"),
    _item("S3", "single", "train", "How many orders were placed in 2024?",
          "SELECT COUNT(*) FROM orders WHERE strftime('%Y', order_date) = '2024';"),
    _item("S4", "single", "train",
          "Which product categories contain more than 3 products? Return the category and its product count.",
          "SELECT category, COUNT(*) AS product_count FROM products GROUP BY category HAVING COUNT(*) > 3;"),
    _item("S5", "single", "train", "What is the average price of products in the Electronics category?",
          "SELECT AVG(price) FROM products WHERE category = 'Electronics';"),
    _item("S6", "single", "train", "How many orders have the status 'pending'?",
          "SELECT COUNT(*) FROM orders WHERE status = 'pending';"),
    _item("S7", "single", "train", "How many customers signed up in 2022?",
          "SELECT COUNT(*) FROM customers WHERE strftime('%Y', signup_date) = '2022';"),
    _item("S8", "single", "train", "What is the total quantity of items across all order lines?",
          "SELECT SUM(quantity) FROM order_items;"),
    _item("S9", "single", "train", "List the names of products priced between 20 and 40 dollars, inclusive.",
          "SELECT name FROM products WHERE price BETWEEN 20 AND 40;"),
    _item("S10", "single", "train", "How many different countries do our customers come from?",
          "SELECT COUNT(DISTINCT country) FROM customers;"),
    _item("S11", "single", "train",
          "Which orders contain more than 2 different products? Return the order id and the number of distinct products.",
          "SELECT order_id, COUNT(DISTINCT product_id) FROM order_items GROUP BY order_id "
          "HAVING COUNT(DISTINCT product_id) > 2;"),
    _item("S12", "single", "train", "What is the average quantity per order line, rounded to 2 decimals?",
          "SELECT ROUND(AVG(quantity), 2) FROM order_items;"),
    _item("S13", "single", "test", "How many orders were placed in March 2024?",
          "SELECT COUNT(*) FROM orders WHERE strftime('%Y-%m', order_date) = '2024-03';"),
    _item("S14", "single", "test",
          "Which countries have at least 7 customers? Return the country and the number of customers.",
          "SELECT country, COUNT(*) AS customer_count FROM customers GROUP BY country HAVING COUNT(*) >= 7;"),
    _item("S15", "single", "test", "List the 3 customers who signed up most recently, with their signup dates.",
          "SELECT name, signup_date FROM customers ORDER BY signup_date DESC LIMIT 3;"),
    _item("S16", "single", "test", "What is the highest product price?",
          "SELECT MAX(price) FROM products;"),
    _item("S17", "single", "test", "List the 5 cheapest products with their prices, cheapest first.",
          "SELECT name, price FROM products ORDER BY price ASC LIMIT 5;"),
    _item("S18", "single", "test", "How many orders were either refunded or cancelled?",
          "SELECT COUNT(*) FROM orders WHERE status IN ('refunded', 'cancelled');"),
    _item("S19", "single", "test",
          "Which products appear in more than 10 order lines? Return the product id and the number of order lines.",
          "SELECT product_id, COUNT(*) FROM order_items GROUP BY product_id HAVING COUNT(*) > 10;"),
    _item("S20", "single", "test", "How many orders were placed in each year? Return the year and the order count.",
          "SELECT strftime('%Y', order_date) AS year, COUNT(*) FROM orders GROUP BY year;"),
    # ================================================================== #
    # Tier 2: multi-join  (M1-M12 train, M13-M20 test)
    # ================================================================== #
    _item("M1", "multi", "train",
          "List each customer's name with the number of orders they have placed. "
          "Include only customers with at least one order.",
          """SELECT c.name, COUNT(o.order_id) AS order_count
             FROM customers c JOIN orders o ON c.customer_id = o.customer_id
             GROUP BY c.customer_id, c.name;"""),
    _item("M2", "multi", "train", "What is the total revenue?",
          f"SELECT SUM(oi.quantity * p.price) AS total_revenue {_ORDER_LINES} WHERE o.status = 'completed';"),
    _item("M3", "multi", "train", "How many different customers from the USA have placed at least one order?",
          """SELECT COUNT(DISTINCT c.customer_id)
             FROM customers c JOIN orders o ON c.customer_id = o.customer_id
             WHERE c.country = 'USA';"""),
    _item("M4", "multi", "train", "List the names of customers who have never placed an order.",
          """SELECT c.name
             FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id
             WHERE o.order_id IS NULL;"""),
    _item("M5", "multi", "train",
          "Which customers have placed more than 3 orders? Return the customer name and the number of orders.",
          """SELECT c.name, COUNT(o.order_id) AS order_count
             FROM customers c JOIN orders o ON c.customer_id = o.customer_id
             GROUP BY c.customer_id, c.name
             HAVING COUNT(o.order_id) > 3;"""),
    _item("M6", "multi", "train", "How many orders were placed by customers from Canada in 2024?",
          """SELECT COUNT(*)
             FROM orders o JOIN customers c ON o.customer_id = c.customer_id
             WHERE c.country = 'Canada' AND strftime('%Y', o.order_date) = '2024';"""),
    _item("M7", "multi", "train",
          "What is the total quantity ordered of each product that has been ordered? "
          "Return the product name and the total quantity.",
          """SELECT p.name, SUM(oi.quantity) AS total_quantity
             FROM products p JOIN order_items oi ON oi.product_id = p.product_id
             GROUP BY p.product_id, p.name;"""),
    _item("M8", "multi", "train", "How many orders did customers from each country place? Return the country and the order count.",
          """SELECT c.country, COUNT(o.order_id) AS order_count
             FROM customers c JOIN orders o ON c.customer_id = o.customer_id
             GROUP BY c.country;"""),
    _item("M9", "multi", "train", "List the names of customers who ordered at least one product in the Toys category.",
          f"""SELECT DISTINCT c.name
             FROM customers c JOIN orders o ON c.customer_id = o.customer_id
             JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id
             WHERE p.category = 'Toys';"""),
    _item("M10", "multi", "train",
          "What is the date of the first completed order of each customer from the USA that has one? "
          "Return the customer name and the date.",
          """SELECT c.name, MIN(o.order_date) AS first_order
             FROM customers c JOIN orders o ON c.customer_id = o.customer_id
             WHERE c.country = 'USA' AND o.status = 'completed'
             GROUP BY c.customer_id, c.name;"""),
    _item("M11", "multi", "train",
          "List every product with the total quantity ordered of it. Products that were never ordered must "
          "appear with a quantity of 0.",
          """SELECT p.name, COALESCE(SUM(oi.quantity), 0) AS total_quantity
             FROM products p LEFT JOIN order_items oi ON oi.product_id = p.product_id
             GROUP BY p.product_id, p.name;"""),
    _item("M12", "multi", "train", "Which customers ordered the product 'Wireless Mouse'? Return their names.",
          """SELECT DISTINCT c.name
             FROM customers c JOIN orders o ON c.customer_id = o.customer_id
             JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id
             WHERE p.name = 'Wireless Mouse';"""),
    _item("M13", "multi", "test",
          "Which 3 product categories generated the most revenue? Return the category and its revenue.",
          f"""SELECT p.category, SUM(oi.quantity * p.price) AS revenue
             {_ORDER_LINES}
             WHERE o.status = 'completed'
             GROUP BY p.category
             ORDER BY revenue DESC
             LIMIT 3;"""),
    _item("M14", "multi", "test",
          "How many orders did each customer from Germany place, including customers who have not ordered? "
          "Return the customer name and the order count.",
          """SELECT c.name, COUNT(o.order_id) AS order_count
             FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id
             WHERE c.country = 'Germany'
             GROUP BY c.customer_id, c.name;"""),
    _item("M15", "multi", "test",
          "What was the total quantity of items sold in each product category in 2024, counting only completed "
          "orders? Return the category and the total quantity.",
          f"""SELECT p.category, SUM(oi.quantity) AS total_quantity
             {_ORDER_LINES}
             WHERE o.status = 'completed' AND strftime('%Y', o.order_date) = '2024'
             GROUP BY p.category;"""),
    _item("M16", "multi", "test", "How many different products have been ordered by customers from the UK?",
          """SELECT COUNT(DISTINCT oi.product_id)
             FROM customers c
             JOIN orders o ON c.customer_id = o.customer_id
             JOIN order_items oi ON o.order_id = oi.order_id
             WHERE c.country = 'UK';"""),
    _item("M17", "multi", "test", "Which product has the highest total quantity sold in completed orders? Return its name.",
          f"""SELECT p.name {_ORDER_LINES}
             WHERE o.status = 'completed'
             GROUP BY p.product_id, p.name
             ORDER BY SUM(oi.quantity) DESC
             LIMIT 1;"""),
    _item("M18", "multi", "test", "How many cancelled orders were placed by customers from India?",
          """SELECT COUNT(*)
             FROM orders o JOIN customers c ON o.customer_id = c.customer_id
             WHERE c.country = 'India' AND o.status = 'cancelled';"""),
    _item("M19", "multi", "test",
          "For each product category, how many different customers have bought from it? "
          "Return the category and the number of customers.",
          f"""SELECT p.category, COUNT(DISTINCT o.customer_id) AS customers
             {_ORDER_LINES}
             GROUP BY p.category;"""),
    _item("M20", "multi", "test",
          "List every product with the number of different orders that contain it. Products that were never "
          "ordered must appear with 0.",
          """SELECT p.name, COUNT(DISTINCT oi.order_id) AS orders_containing
             FROM products p LEFT JOIN order_items oi ON oi.product_id = p.product_id
             GROUP BY p.product_id, p.name;"""),
    # ================================================================== #
    # Tier 3: advanced  (A1-A12 train, A13-A20 test)
    # ================================================================== #
    _item("A1", "advanced", "train",
          "Which customers have spent more than the average customer? Count only completed orders and compare "
          "against the average over customers who have completed orders. Return the name and total spent.",
          f"WITH {_SPEND_CTE} SELECT name, total FROM spend WHERE total > (SELECT AVG(total) FROM spend);"),
    _item("A2", "advanced", "train", "Which products have never been ordered?",
          """SELECT p.name FROM products p
             WHERE NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.product_id = p.product_id);"""),
    _item("A3", "advanced", "train", "What percentage of all orders were cancelled? Round to 1 decimal.",
          "SELECT ROUND(100.0 * SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*), 1) FROM orders;"),
    _item("A4", "advanced", "train",
          "Which customers have at least one order and all of their orders are completed? Return their names.",
          """SELECT c.name
             FROM customers c JOIN orders o ON c.customer_id = o.customer_id
             GROUP BY c.customer_id, c.name
             HAVING COUNT(*) = SUM(CASE WHEN o.status = 'completed' THEN 1 ELSE 0 END);"""),
    _item("A5", "advanced", "train", "What is the average order value, counting only completed orders?",
          f"""SELECT AVG(order_total) FROM (
                 SELECT o.order_id, SUM(oi.quantity * p.price) AS order_total
                 {_ORDER_LINES}
                 WHERE o.status = 'completed' GROUP BY o.order_id);"""),
    _item("A6", "advanced", "train",
          "Rank the product categories by total quantity sold in completed orders, using RANK(). "
          "Return the category, the quantity and the rank.",
          f"""SELECT category, qty, RANK() OVER (ORDER BY qty DESC) AS rnk FROM (
                 SELECT p.category AS category, SUM(oi.quantity) AS qty
                 {_ORDER_LINES}
                 WHERE o.status = 'completed' GROUP BY p.category);"""),
    _item("A7", "advanced", "train", "How many customers placed orders in both 2023 and 2024?",
          """SELECT COUNT(*) FROM (
                 SELECT customer_id FROM orders
                 WHERE strftime('%Y', order_date) IN ('2023', '2024')
                 GROUP BY customer_id
                 HAVING COUNT(DISTINCT strftime('%Y', order_date)) = 2);"""),
    _item("A8", "advanced", "train",
          "Which product categories have an average product price above the overall average product price?",
          "SELECT category FROM products GROUP BY category HAVING AVG(price) > (SELECT AVG(price) FROM products);"),
    _item("A9", "advanced", "train",
          "What percentage of customers have placed at least one order? Round to 1 decimal.",
          "SELECT ROUND(100.0 * COUNT(DISTINCT o.customer_id) / (SELECT COUNT(*) FROM customers), 1) FROM orders o;"),
    _item("A10", "advanced", "train", "Which customers have ordered products from every product category?",
          f"""SELECT c.name
             FROM customers c JOIN orders o ON c.customer_id = o.customer_id
             JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id
             GROUP BY c.customer_id, c.name
             HAVING COUNT(DISTINCT p.category) = (SELECT COUNT(DISTINCT category) FROM products);"""),
    _item("A11", "advanced", "train",
          "Show the top 2 customers by completed-order spend within each country. "
          "Return the country, the customer name and the total spent.",
          f"""WITH {_SPEND_CTE},
             ranked AS (SELECT country, name, total,
                               ROW_NUMBER() OVER (PARTITION BY country ORDER BY total DESC) AS rn FROM spend)
             SELECT country, name, total FROM ranked WHERE rn <= 2;"""),
    _item("A12", "advanced", "train", "Which products are priced above the average price of their own category?",
          """SELECT p.name FROM products p
             WHERE p.price > (SELECT AVG(p2.price) FROM products p2 WHERE p2.category = p.category);"""),
    _item("A13", "advanced", "test",
          "Which customers have placed more orders than the average number of orders per customer who has "
          "ordered? Return their names.",
          """WITH counts AS (SELECT c.customer_id, c.name, COUNT(o.order_id) AS n
                            FROM customers c JOIN orders o ON c.customer_id = o.customer_id
                            GROUP BY c.customer_id, c.name)
             SELECT name FROM counts WHERE n > (SELECT AVG(n) FROM counts);"""),
    _item("A14", "advanced", "test", "For each product category, which is the most expensive product? Return the category, the product name and its price.",
          """SELECT category, name, price FROM (
                 SELECT category, name, price,
                        ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC) AS rn FROM products)
             WHERE rn = 1;"""),
    _item("A15", "advanced", "test",
          "In which month of 2024 were the most orders placed? Return the month as YYYY-MM and the order count.",
          """WITH monthly AS (SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS n FROM orders
                              WHERE strftime('%Y', order_date) = '2024' GROUP BY month)
             SELECT month, n FROM monthly WHERE n = (SELECT MAX(n) FROM monthly);"""),
    _item("A16", "advanced", "test", "How many customers have no completed order at all (including customers who never ordered)?",
          """SELECT COUNT(*) FROM customers c
             WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id AND o.status = 'completed');"""),
    _item("A17", "advanced", "test",
          "What share of the total completed-order revenue comes from the Electronics category? "
          "Return a percentage rounded to 1 decimal.",
          f"""SELECT ROUND(100.0 * SUM(CASE WHEN p.category = 'Electronics' THEN oi.quantity * p.price ELSE 0 END)
                          / SUM(oi.quantity * p.price), 1)
             {_ORDER_LINES} WHERE o.status = 'completed';"""),
    _item("A18", "advanced", "test",
          "Which orders have a total value above the average order total (over all orders, any status)? "
          "Return the order id and its total.",
          f"""WITH totals AS (SELECT o.order_id, SUM(oi.quantity * p.price) AS total
                              {_ORDER_LINES} GROUP BY o.order_id)
             SELECT order_id, total FROM totals WHERE total > (SELECT AVG(total) FROM totals);"""),
    _item("A19", "advanced", "test",
          "Which customers have ordered more different products than the average customer who has ordered? "
          "Return their names.",
          """WITH per_customer AS (SELECT c.customer_id, c.name, COUNT(DISTINCT oi.product_id) AS n
                                  FROM customers c JOIN orders o ON c.customer_id = o.customer_id
                                  JOIN order_items oi ON o.order_id = oi.order_id
                                  GROUP BY c.customer_id, c.name)
             SELECT name FROM per_customer WHERE n > (SELECT AVG(n) FROM per_customer);"""),
    _item("A20", "advanced", "test", "For each country, which customer signed up first? Return the country and the customer name.",
          """SELECT country, name FROM (
                 SELECT country, name,
                        ROW_NUMBER() OVER (PARTITION BY country ORDER BY signup_date) AS rn FROM customers)
             WHERE rn = 1;"""),
]


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #


def normalize_question(question: str) -> str:
    """Lower-case, collapse whitespace and drop trailing punctuation."""
    return re.sub(r"\s+", " ", question or "").strip().lower().rstrip("?.! ")


def items_for(tier: str, split: str | None = None, limit: int | None = None) -> list[BenchmarkItem]:
    """Benchmark items of a tier (file order), optionally one split and/or only the first ``limit``."""
    found = [i for i in BENCHMARK if i.tier == tier and (split is None or i.split == split)]
    return found[:limit] if limit is not None else found


def find_item(question: str) -> BenchmarkItem | None:
    """Exact (normalised) match of a question against the benchmark."""
    key = normalize_question(question)
    return next((i for i in BENCHMARK if normalize_question(i.question) == key), None)


def find_item_fuzzy(question: str, cutoff: float = 0.82) -> BenchmarkItem | None:
    """Like ``find_item`` but tolerates small typos / rewordings."""
    exact = find_item(question)
    if exact is not None:
        return exact
    keys = {normalize_question(i.question): i for i in BENCHMARK}
    close = difflib.get_close_matches(normalize_question(question), list(keys), n=1, cutoff=cutoff)
    return keys[close[0]] if close else None


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #

_ADVANCED_RE = re.compile(r"\b(with|over|case|exists)\b|\(\s*select\b", re.IGNORECASE)


def validate_benchmark() -> list[str]:
    """Return a list of problems (empty list == benchmark is sound)."""
    import database as db

    db.init_database()
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for item in BENCHMARK:
        if item.id in seen_ids:
            problems.append(f"{item.id}: duplicate id")
        seen_ids.add(item.id)
        if normalize_question(item.question) in seen_questions:
            problems.append(f"{item.id}: duplicate question")
        seen_questions.add(normalize_question(item.question))

        result = db.run_sql(item.gold_sql)
        if not result.ok:
            problems.append(f"{item.id}: gold SQL fails: {result.error}")
            continue
        if result.row_count == 0:
            problems.append(f"{item.id}: gold SQL returns no rows (too easy to game)")

        has_join = re.search(r"\bjoin\b", item.gold_sql, re.IGNORECASE) is not None
        fancy = _ADVANCED_RE.search(item.gold_sql) is not None
        if item.tier == "single" and (has_join or fancy):
            problems.append(f"{item.id}: single-table item uses a JOIN or advanced construct")
        if item.tier == "multi" and (not has_join or fancy):
            problems.append(f"{item.id}: multi-join item must JOIN and stay free of subqueries/CTEs/windows")
        if item.tier == "advanced" and not fancy:
            problems.append(f"{item.id}: advanced item uses no subquery, CTE, window or CASE")
    for tier in TIER_LABELS:
        for split, expected in (("train", 12), ("test", 8)):
            n = len(items_for(tier, split))
            if n != expected:
                problems.append(f"tier {tier!r} has {n} {split} items, expected {expected}")
    return problems


if __name__ == "__main__":
    issues = validate_benchmark()
    for tier, label in TIER_LABELS.items():
        print(f"{label}: {len(items_for(tier, 'train'))} train / {len(items_for(tier, 'test'))} test questions")
    if issues:
        print("\nProblems found:")
        for issue in issues:
            print(" -", issue)
        raise SystemExit(1)
    print(f"\nAll {len(BENCHMARK)} gold queries execute and every tier is well-formed.")
