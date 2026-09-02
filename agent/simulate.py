"""
Runs the trained Q-learning agent AND every baseline over the same
synthetic transaction set (data/failed_payments.json), producing:
  - agent/results/metrics.json        (aggregate comparison, for the pitch)
  - agent/results/audit_log.json      (per-transaction decision trail)

Each (transaction, policy) pair gets its own deterministic RNG seed so
results are reproducible across runs, while different policies are free
to diverge in which actions (and therefore which random draws) they make.
"""
import hashlib
import json
import random

from agent import environment as env
from agent import guardrails as gr
from agent.baselines import BASELINES
from agent.policy import QLearningPolicy
from agent.state import make_state


def seeded_rng(transaction_id, policy_name):
    h = hashlib.sha256(f"{transaction_id}:{policy_name}".encode()).hexdigest()
    return random.Random(int(h[:8], 16))


def run_transaction(txn, decide_fn, rng):
    failure_reason = txn["failure_reason"]
    amount = txn["amount_inr"]
    amt_bucket = txn["amount_bucket"]
    risk = txn["customer_risk"]
    hour_of_day = rng.randint(0, 23)

    attempt_number = 0
    hours_since_failure = 0.0
    trace = []

    while True:
        state = make_state(failure_reason, amt_bucket, attempt_number, risk, hours_since_failure)
        valid = gr.valid_actions(attempt_number, hours_since_failure, amt_bucket, risk, hour_of_day)
        action = decide_fn(state, valid, rng)
        assert action in valid, f"guardrail violation: {action} not in {valid}"

        if action == "stop":
            trace.append({"attempt": attempt_number, "action": "stop", "success": False,
                           "reward": 0.0, "cost": 0.0})
            break

        success, reward = env.step(failure_reason, action, attempt_number, risk, amount, rng)
        cost = (amount - reward) if success else (-reward)
        trace.append({"attempt": attempt_number, "action": action, "success": success,
                       "reward": round(reward, 2), "cost": round(cost, 2)})

        attempt_number += 1
        hours_since_failure += gr.next_cooldown_hours(action)
        hour_of_day = (hour_of_day + gr.next_cooldown_hours(action)) % 24

        if success or attempt_number >= gr.MAX_ATTEMPTS or hours_since_failure >= gr.FORCED_STOP_HOURS:
            break

    recovered = any(t["success"] for t in trace)
    total_cost = sum(t["cost"] for t in trace)
    escalations = [t for t in trace if t["action"] == "escalate_human"]
    false_escalation = bool(escalations) and not recovered

    return {
        "transaction_id": txn["transaction_id"],
        "customer_id": txn["customer_id"],
        "amount_inr": amount,
        "failure_reason": failure_reason,
        "customer_risk": risk,
        "recovered": recovered,
        "revenue_recovered": amount if recovered else 0.0,
        "total_cost": round(total_cost, 2),
        "net_value": round((amount if recovered else 0.0) - total_cost, 2),
        "attempts": len(trace),
        "escalated": bool(escalations),
        "false_escalation": false_escalation,
        "trace": trace,
    }


def summarize(records):
    n = len(records)
    revenue = sum(r["revenue_recovered"] for r in records)
    cost = sum(r["total_cost"] for r in records)
    recovered_count = sum(1 for r in records if r["recovered"])
    escalations = sum(1 for r in records if r["escalated"])
    false_escalations = sum(1 for r in records if r["false_escalation"])
    guardrail_violations = 0  # enforced by assert in run_transaction — always 0 by construction
    avg_attempts = sum(r["attempts"] for r in records) / n if n else 0

    return {
        "transactions": n,
        "recovery_rate": round(recovered_count / n, 4) if n else 0,
        "revenue_recovered_inr": round(revenue, 2),
        "operating_cost_inr": round(cost, 2),
        "net_revenue_recovered_inr": round(revenue - cost, 2),
        "escalations": escalations,
        "false_escalation_rate": round(false_escalations / escalations, 4) if escalations else 0.0,
        "guardrail_violations": guardrail_violations,
        "avg_attempts_per_transaction": round(avg_attempts, 2),
    }


def main():
    with open("data/failed_payments.json") as f:
        transactions = json.load(f)

    policy = QLearningPolicy()
    policy.load("agent/q_table.json")

    def agent_decide(state, valid, rng):
        return policy.act_greedy(state, valid)

    decision_fns = {"reclaimai_agent": agent_decide, **BASELINES}

    all_metrics = {}
    audit_log = {}

    for name, fn in decision_fns.items():
        records = []
        for txn in transactions:
            rng = seeded_rng(txn["transaction_id"], name)
            records.append(run_transaction(txn, fn, rng))
        all_metrics[name] = summarize(records)
        audit_log[name] = records

    import os
    os.makedirs("agent/results", exist_ok=True)
    with open("agent/results/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open("agent/results/audit_log.json", "w") as f:
        json.dump(audit_log, f, indent=2)

    print(json.dumps(all_metrics, indent=2))

    baseline_best = max(
        (all_metrics[b]["net_revenue_recovered_inr"] for b in BASELINES),
        default=0,
    )
    agent_net = all_metrics["reclaimai_agent"]["net_revenue_recovered_inr"]
    if baseline_best:
        lift = (agent_net - baseline_best) / abs(baseline_best) * 100
        print(f"\nAgent beats best baseline net revenue by {lift:.1f}%")


if __name__ == "__main__":
    main()
