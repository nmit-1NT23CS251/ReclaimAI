"""
Sensitivity / robustness check.

The honest weak point of agent/simulate.py is that the same person (me)
wrote both the agent and the simulated environment it's graded against —
so "the agent beats baselines by 32%" only proves something if that
finding isn't just an artifact of one convenient set of made-up
probability/cost numbers.

This script re-runs the identical agent-vs-baseline comparison several
times, each time jittering every recovery-probability and cost assumption
in agent/environment.py by up to +/-25%, and checks whether the agent
still wins. If it wins across a spread of perturbed assumptions, that's
real evidence the *qualitative* finding (context-aware decisions beat
context-blind ones) is robust — even though the exact rupee figures
remain illustrative, not a production forecast.
"""
import copy
import json
import random

from agent import environment as env
from agent.baselines import BASELINES
from agent.policy import QLearningPolicy
from agent.simulate import run_transaction, seeded_rng, summarize

TRIALS = 8
JITTER = 0.25  # +/- 25% on every probability and cost assumption


def perturb(rng):
    """Mutates agent.environment's module-level assumption tables in place
    and returns the originals so they can be restored afterward."""
    saved = (
        copy.deepcopy(env.RECOVERY_BASE_PROB),
        copy.deepcopy(env.FLAT_COST),
        env.DISCOUNT_RATE,
    )

    env.RECOVERY_BASE_PROB = {
        reason: {a: max(0.01, min(0.95, p * rng.uniform(1 - JITTER, 1 + JITTER)))
                  for a, p in actions.items()}
        for reason, actions in saved[0].items()
    }
    env.FLAT_COST = {a: max(0.0, c * rng.uniform(1 - JITTER, 1 + JITTER)) for a, c in saved[1].items()}
    env.DISCOUNT_RATE = max(0.02, saved[2] * rng.uniform(1 - JITTER, 1 + JITTER))

    return saved


def restore(saved):
    env.RECOVERY_BASE_PROB, env.FLAT_COST, env.DISCOUNT_RATE = saved


def run_one_trial(transactions, decision_fns, trial_idx):
    trial_metrics = {}
    for name, fn in decision_fns.items():
        records = [
            run_transaction(t, fn, seeded_rng(t["transaction_id"], f"{name}-robustness{trial_idx}"))
            for t in transactions
        ]
        trial_metrics[name] = summarize(records)
    return trial_metrics


def main():
    with open("data/failed_payments.json") as f:
        transactions = json.load(f)

    policy = QLearningPolicy()
    policy.load("agent/q_table.json")

    def agent_decide(state, valid, rng):
        return policy.act_greedy(state, valid)

    decision_fns = {"reclaimai_agent": agent_decide, **BASELINES}

    master_rng = random.Random(2024)
    results = []

    for trial in range(TRIALS):
        trial_rng = random.Random(master_rng.randint(0, 2**31))
        saved = perturb(trial_rng)
        try:
            metrics = run_one_trial(transactions, decision_fns, trial)
        finally:
            restore(saved)

        best_baseline_net = max(metrics[b]["net_revenue_recovered_inr"] for b in BASELINES)
        agent_net = metrics["reclaimai_agent"]["net_revenue_recovered_inr"]
        lift_pct = (agent_net - best_baseline_net) / abs(best_baseline_net) * 100 if best_baseline_net else float("inf")
        agent_wins = agent_net > best_baseline_net

        results.append({
            "trial": trial, "agent_net_inr": round(agent_net, 2),
            "best_baseline_net_inr": round(best_baseline_net, 2),
            "lift_pct": round(lift_pct, 1), "agent_wins": agent_wins,
        })
        print(f"trial {trial}: agent net {round(agent_net):,} vs best baseline {round(best_baseline_net):,}  "
              f"({'WIN' if agent_wins else 'LOSS'}, lift {lift_pct:+.1f}%)")

    with open("agent/results/robustness.json", "w") as f:
        json.dump(results, f, indent=2)

    wins = sum(1 for r in results if r["agent_wins"])
    lifts = [r["lift_pct"] for r in results]
    print(f"\nAcross {TRIALS} trials with every probability/cost assumption jittered +/-{int(JITTER*100)}%:")
    print(f"  agent beat the best baseline in {wins}/{TRIALS} trials")
    print(f"  lift ranged {min(lifts):+.1f}% to {max(lifts):+.1f}%  (avg {sum(lifts)/len(lifts):+.1f}%)")


if __name__ == "__main__":
    main()
