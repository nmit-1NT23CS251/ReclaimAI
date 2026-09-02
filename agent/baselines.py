"""Baseline strategies the trained agent is measured against. All three
still obey the guardrails module — the comparison is about decision
quality, not about whether they're allowed to spam."""
import random


def always_retry_now(state, valid, rng: random.Random):
    return "retry_now" if "retry_now" in valid else "stop"


def retry_then_stop(state, valid, rng: random.Random):
    attempt_bucket = state[2]
    if attempt_bucket == 0 and "retry_now" in valid:
        return "retry_now"
    return "stop"


def random_valid(state, valid, rng: random.Random):
    return rng.choice(valid)


BASELINES = {
    "always_retry_now": always_retry_now,
    "retry_then_stop": retry_then_stop,
    "random_valid": random_valid,
}
