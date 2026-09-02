"""
Generates a synthetic stream of failed-payment / checkout-abandonment events,
the way they'd arrive from a payment gateway + checkout funnel.

Not meant to mimic any real merchant's actual data — purely synthetic,
seeded for reproducibility, sized to be a meaningful eval set (500+ events)
per the Finance/Risk/Revenue-Recovery track's "50+ synthetic records" floor.
"""
import csv
import json
import random
from datetime import datetime, timedelta

random.seed(42)

FAILURE_REASONS = [
    ("insufficient_funds", 0.32),
    ("bank_timeout", 0.18),
    ("risk_decline", 0.12),
    ("otp_failed", 0.16),
    ("card_expired", 0.08),
    ("cart_abandoned", 0.14),
]

CUSTOMER_RISK = [("low", 0.70), ("medium", 0.22), ("high", 0.08)]


def weighted_choice(pairs):
    r = random.random()
    upto = 0.0
    for value, weight in pairs:
        upto += weight
        if r <= upto:
            return value
    return pairs[-1][0]


def amount_bucket(amount):
    if amount < 500:
        return "low"
    if amount <= 5000:
        return "mid"
    return "high"


def generate_events(n=600, start=None):
    start = start or datetime(2026, 8, 1, 0, 0, 0)
    events = []
    for i in range(n):
        failure_reason = weighted_choice(FAILURE_REASONS)
        risk = weighted_choice(CUSTOMER_RISK)

        # amount distribution: log-normal-ish via random skew, in INR
        base = random.choice([random.uniform(50, 500), random.uniform(500, 5000),
                               random.uniform(5000, 50000)])
        amount = round(base, 2)

        ts = start + timedelta(minutes=random.randint(0, 60 * 24 * 21))  # 3-week window

        events.append({
            "transaction_id": f"txn_{i:05d}",
            "amount_inr": amount,
            "amount_bucket": amount_bucket(amount),
            "failure_reason": failure_reason,
            "customer_risk": risk,
            "failed_at": ts.isoformat(),
            "customer_id": f"cust_{random.randint(1, 400):04d}",
        })
    return events


def main():
    events = generate_events(600)
    with open("data/failed_payments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        writer.writeheader()
        writer.writerows(events)
    with open("data/failed_payments.json", "w") as f:
        json.dump(events, f, indent=2)
    print(f"Generated {len(events)} synthetic failed-payment/abandonment events.")


if __name__ == "__main__":
    main()
