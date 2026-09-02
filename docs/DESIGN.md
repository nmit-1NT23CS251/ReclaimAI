# ReclaimAI — Design Notes

## Problem
Merchants lose revenue two ways: (1) payments that fail at gateway/bank level
(insufficient funds, bank timeout, risk decline, wrong OTP) and (2) checkout
abandonment (cart built, payment never attempted). Most merchants either do
nothing, or blast every failure with the same retry/email regardless of
context — wasting goodwill, tripping spam/compliance limits, and leaving
recoverable revenue on the table.

ReclaimAI is an agent that decides, **per failed/abandoned transaction**,
the single best next action — and knows when to stop.

## State space (per transaction, per decision step)
- `failure_reason`: insufficient_funds | bank_timeout | risk_decline | otp_failed | card_expired | cart_abandoned
- `amount_bucket`: low (<₹500) | mid (₹500–5000) | high (>₹5000)
- `attempt_number`: 0..4 (how many recovery actions already taken)
- `customer_risk`: low | medium | high (synthetic proxy for chargeback/fraud history)
- `hours_since_failure`: bucketed (0-1h, 1-6h, 6-24h, 24h+)

## Action space
- `retry_now` — silent gateway retry, no customer contact
- `retry_with_reminder` — notification + retry link
- `offer_discount` — small incentive (only viable above a margin floor) + retry link
- `escalate_human` — hand off to support/ops queue
- `stop` — write off / no further action

## Guardrails (hard-coded, agent cannot override)
- Max **4** recovery actions per transaction, ever.
- Minimum **2h cooldown** between contacts (no spam).
- No customer contact 10pm–8am ("quiet hours").
- `escalate_human` only for `amount_bucket == high` or `customer_risk == high`
  (keeps human ops queue small and load-bearing).
- Mandatory `stop` after 72h with no recovery — write-off, logged.
- Every decision is written to an immutable audit log (reason code + inputs +
  action + human-readable explanation), before the action is "executed."

## Policy
Q-learning table over the discretized state above, trained inside a
simulated environment that models recovery probability per
(failure_reason, action, attempt_number) plus a cost per action (discount
cost, support-ops cost, notification fatigue that lowers future recovery
odds). Reward = revenue recovered − action cost − fatigue penalty. This
mirrors the Q-table approach already proven on the SafarAI RL backend
(Bellman update, epsilon-greedy exploration during training, greedy at
inference), just applied to a new environment.

Baselines the agent is measured against:
1. `always_retry_now` — naive, no discounts/escalation, no stopping discipline
2. `retry_then_stop` — one retry, then give up
3. `random_valid_action` — random pick within guardrails (sanity floor)

## Explanation layer
Every agent decision is fed to an LLM (Groq/Llama, same provider used in
SafarAI's Destination Guide) to produce: (a) a one-line human-readable
rationale for the audit log, and (b) a customer-facing recovery message
when the action involves contact. If no `GROQ_API_KEY` is set, a templated
fallback produces equivalent (if less fluent) text so the whole system
still runs end-to-end offline/without a key.

## Metrics reported
- Total simulated revenue recovered, agent vs each baseline (₹ and %)
- False-escalation rate (escalations that recover nothing / low-value)
- Guardrail violations = 0 (proof by construction — asserted in tests)
- Average actions-to-recovery
