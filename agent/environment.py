"""
Simulated recovery environment. This is a model of merchant reality, not
real Razorpay data — it exists so the Q-learning policy has something to
learn against, and so the eval numbers in the pitch are reproducible.

Success probabilities and costs are deliberately explicit constants (not
hidden inside magic numbers) so a reviewer can sanity-check the assumptions
in about 30 seconds.
"""
import random

from agent.guardrails import ACTIONS

RECOVERY_BASE_PROB = {
    "insufficient_funds": {"retry_now": 0.25, "retry_with_reminder": 0.35, "offer_discount": 0.45, "escalate_human": 0.50},
    "bank_timeout":       {"retry_now": 0.55, "retry_with_reminder": 0.50, "offer_discount": 0.50, "escalate_human": 0.55},
    "risk_decline":       {"retry_now": 0.05, "retry_with_reminder": 0.08, "offer_discount": 0.10, "escalate_human": 0.35},
    "otp_failed":         {"retry_now": 0.40, "retry_with_reminder": 0.55, "offer_discount": 0.50, "escalate_human": 0.60},
    "card_expired":       {"retry_now": 0.02, "retry_with_reminder": 0.30, "offer_discount": 0.35, "escalate_human": 0.45},
    "cart_abandoned":     {"retry_now": 0.05, "retry_with_reminder": 0.25, "offer_discount": 0.40, "escalate_human": 0.30},
}

RISK_MULTIPLIER = {"low": 1.0, "medium": 0.9, "high": 0.7}

# flat/variable cost per action, in INR
FLAT_COST = {"retry_now": 1, "retry_with_reminder": 3, "offer_discount": 3, "escalate_human": 60, "stop": 0}
DISCOUNT_RATE = 0.08  # % of transaction value, only charged on success


def success_probability(failure_reason, action, attempt_number, customer_risk):
    base = RECOVERY_BASE_PROB[failure_reason].get(action, 0.0)
    fatigue_penalty = 0.04 * attempt_number
    prob = base * RISK_MULTIPLIER[customer_risk] - fatigue_penalty
    return max(0.01, min(0.95, prob))


def action_cost(action, amount, success):
    cost = FLAT_COST[action]
    if action == "offer_discount" and success:
        cost += DISCOUNT_RATE * amount
    return cost


def step(failure_reason, action, attempt_number, customer_risk, amount, rng: random.Random):
    """Simulate one recovery attempt. Returns (success: bool, reward: float)."""
    if action == "stop":
        return False, 0.0
    prob = success_probability(failure_reason, action, attempt_number, customer_risk)
    success = rng.random() < prob
    cost = action_cost(action, amount, success)
    reward = (amount - cost) if success else -cost
    return success, reward


assert set(FLAT_COST) == set(ACTIONS)
