"""
Hard compliance guardrails. These are NOT learned — the Q-learning policy
can only ever choose among the actions this module says are valid for a
given state. This is what makes the agent "bounded and gated" rather than
a black box that could spam or over-escalate.
"""

ACTIONS = ["retry_now", "retry_with_reminder", "offer_discount", "escalate_human", "stop"]

MAX_ATTEMPTS = 4
FORCED_STOP_HOURS = 72
QUIET_HOURS_START = 22  # 10pm
QUIET_HOURS_END = 8     # 8am
COOLDOWN_HOURS_BY_ACTION = {
    "retry_now": 1,
    "retry_with_reminder": 4,
    "offer_discount": 6,
    "escalate_human": 24,
    "stop": 0,
}


def is_quiet_hours(hour_of_day: int) -> bool:
    return hour_of_day >= QUIET_HOURS_START or hour_of_day < QUIET_HOURS_END


def valid_actions(attempt_number, hours_since_failure, amount_bucket, customer_risk, hour_of_day):
    """Returns the subset of ACTIONS the agent is permitted to choose from.

    Every rule here traces to a stated guardrail in docs/DESIGN.md:
      - hard cap on total attempts
      - mandatory write-off after 72h
      - escalation reserved for high-value / high-risk cases only
      - no customer-facing contact during quiet hours
    """
    if attempt_number >= MAX_ATTEMPTS:
        return ["stop"]
    if hours_since_failure >= FORCED_STOP_HOURS:
        return ["stop"]

    valid = ["retry_now", "stop"]  # always allowed: silent retry, or give up

    contact_allowed = not is_quiet_hours(hour_of_day)
    if contact_allowed:
        valid.append("retry_with_reminder")
        if attempt_number >= 1:  # never lead with a discount — try cheaper options first
            valid.append("offer_discount")
        if amount_bucket == "high" or customer_risk == "high":
            valid.append("escalate_human")

    return valid


def next_cooldown_hours(action: str) -> int:
    return COOLDOWN_HOURS_BY_ACTION.get(action, 1)
