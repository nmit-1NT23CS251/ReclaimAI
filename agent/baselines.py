"""Baseline strategies the trained agent is measured against. All four
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


def smart_rules(state, valid, rng: random.Random):
    """A hand-crafted, non-learned policy — the kind a reasonably thoughtful
    product manager might write as if/else logic without any ML. This is the
    baseline that actually matters: beating "always retry blindly" or
    "random" is a low bar, but beating a sensible hand-written rule set is
    real evidence the learned policy is adding value, not just avoiding
    obviously dumb behavior.

    The rules, in order:
      1. Once a cheap first attempt has been made on a big/risky transaction,
         hand it to a human — don't keep messaging automatically.
      2. On the very first attempt, only retry silently for failure types
         that plausibly self-resolve (bank hiccup, temporary insufficient
         funds, a mistyped OTP); everything else gets a reminder instead of
         a wasted silent retry.
      3. On later attempts, escalate through reminder -> discount -> silent
         retry -> give up, whichever is the cheapest still-valid option.
    """
    failure_reason, amount_bucket, attempt_bucket, customer_risk, hours_bucket = state

    if attempt_bucket >= 1 and "escalate_human" in valid:
        return "escalate_human"

    if attempt_bucket == 0:
        if failure_reason in ("bank_timeout", "insufficient_funds", "otp_failed") and "retry_now" in valid:
            return "retry_now"
        if "retry_with_reminder" in valid:
            return "retry_with_reminder"

    if "retry_with_reminder" in valid:
        return "retry_with_reminder"
    if "offer_discount" in valid:
        return "offer_discount"
    if "retry_now" in valid:
        return "retry_now"
    return "stop"


BASELINES = {
    "always_retry_now": always_retry_now,
    "retry_then_stop": retry_then_stop,
    "random_valid": random_valid,
    "smart_rules": smart_rules,
}
