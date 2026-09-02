"""Shared state discretization so training, simulation, and the API all
bucket the world the same way."""

FAILURE_REASONS = ["insufficient_funds", "bank_timeout", "risk_decline",
                    "otp_failed", "card_expired", "cart_abandoned"]
AMOUNT_BUCKETS = ["low", "mid", "high"]
RISK_LEVELS = ["low", "medium", "high"]


def hours_bucket(hours_since_failure: float) -> int:
    if hours_since_failure < 1:
        return 0
    if hours_since_failure < 6:
        return 1
    if hours_since_failure < 24:
        return 2
    return 3


def attempt_bucket(attempt_number: int) -> int:
    return min(attempt_number, 3)


def make_state(failure_reason, amount_bucket, attempt_number, customer_risk, hours_since_failure):
    return (
        failure_reason,
        amount_bucket,
        attempt_bucket(attempt_number),
        customer_risk,
        hours_bucket(hours_since_failure),
    )


def state_key(state_tuple) -> str:
    """Q-table is persisted as JSON, so keys must be strings."""
    return "|".join(str(x) for x in state_tuple)
