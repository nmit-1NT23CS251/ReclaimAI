"""SQLite-backed, append-only audit trail for live /decide calls.
(Batch simulation results are read straight from agent/results/*.json —
this DB captures the live, on-demand decisions made through the API.)"""
import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "audit_trail.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    transaction_id TEXT,
    failure_reason TEXT,
    amount_bucket TEXT,
    amount REAL,
    customer_risk TEXT,
    attempt_number INTEGER,
    hours_since_failure REAL,
    action TEXT,
    valid_actions TEXT,
    explanation TEXT,
    customer_message TEXT
);
"""


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()


def log_decision(record: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO audit_log
           (ts, transaction_id, failure_reason, amount_bucket, amount, customer_risk,
            attempt_number, hours_since_failure, action, valid_actions, explanation, customer_message)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            time.time(), record.get("transaction_id"), record["failure_reason"],
            record["amount_bucket"], record["amount"], record["customer_risk"],
            record["attempt_number"], record["hours_since_failure"], record["action"],
            json.dumps(record["valid_actions"]), record["explanation"], record.get("customer_message", ""),
        ),
    )
    conn.commit()
    conn.close()


def recent_decisions(limit=100):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
