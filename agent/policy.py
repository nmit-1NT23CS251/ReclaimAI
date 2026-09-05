"""
Tabular Q-learning policy — same family of technique as the Q-table RL
backend used in SafarAI (Bellman update, epsilon-greedy during training,
greedy at inference), applied here to the recovery-action decision.
"""
import json
import random

from agent import environment as env
from agent import guardrails as gr
from agent.state import make_state, state_key

ALPHA = 0.15   # learning rate
GAMMA = 0.90   # discount factor
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 200000


class QLearningPolicy:
    def __init__(self):
        self.q = {}  # state_key -> {action: value}

    def _ensure(self, key, valid):
        if key not in self.q:
            self.q[key] = {a: 0.0 for a in valid}
        else:
            for a in valid:
                self.q[key].setdefault(a, 0.0)

    def act_epsilon_greedy(self, state, valid, epsilon, rng: random.Random):
        key = state_key(state)
        self._ensure(key, valid)
        if rng.random() < epsilon:
            return rng.choice(valid)
        qvals = self.q[key]
        return max(valid, key=lambda a: qvals.get(a, 0.0))

    def act_greedy(self, state, valid):
        key = state_key(state)
        if key not in self.q:
            # unseen state: fall back to the guardrail-safe default —
            # cheapest non-committal action available.
            for preferred in ("retry_now", "stop"):
                if preferred in valid:
                    return preferred
            return valid[0]
        qvals = self.q[key]
        return max(valid, key=lambda a: qvals.get(a, float("-inf")) if a in qvals else float("-inf"))

    def q_values(self, state, valid):
        """Returns {action: estimated_value or None} for every valid action —
        None means this exact state was never seen during training, so there's
        no learned estimate for it (used to be honest in the UI rather than
        pretend confidence that isn't there)."""
        key = state_key(state)
        if key not in self.q:
            return {a: None for a in valid}
        return {a: self.q[key].get(a) for a in valid}

    def update(self, state, action, reward, next_state, next_valid, done):
        key = state_key(state)
        self._ensure(key, [action])
        if done or next_state is None:
            target = reward
        else:
            nkey = state_key(next_state)
            self._ensure(nkey, next_valid)
            target = reward + GAMMA * max(self.q[nkey].get(a, 0.0) for a in next_valid)
        self.q[key][action] += ALPHA * (target - self.q[key][action])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.q, f)

    def load(self, path):
        with open(path) as f:
            self.q = json.load(f)


FAILURE_REASON_WEIGHTS = [("insufficient_funds", 0.32), ("bank_timeout", 0.18), ("risk_decline", 0.12),
                           ("otp_failed", 0.16), ("card_expired", 0.08), ("cart_abandoned", 0.14)]
CUSTOMER_RISK_WEIGHTS = [("low", 0.70), ("medium", 0.22), ("high", 0.08)]


def _amount_bucket(amount):
    if amount < 500:
        return "low"
    if amount <= 5000:
        return "mid"
    return "high"


def sample_transaction(rng: random.Random):
    failure_reason = rng.choices([f for f, _ in FAILURE_REASON_WEIGHTS], weights=[w for _, w in FAILURE_REASON_WEIGHTS])[0]
    risk = rng.choices([r for r, _ in CUSTOMER_RISK_WEIGHTS], weights=[w for _, w in CUSTOMER_RISK_WEIGHTS])[0]
    amount = round(rng.choice([rng.uniform(50, 500), rng.uniform(500, 5000), rng.uniform(5000, 50000)]), 2)
    return failure_reason, amount, _amount_bucket(amount), risk


def run_episode(policy: QLearningPolicy, rng: random.Random, training: bool, epsilon: float = 0.0):
    failure_reason, amount, amt_bucket, risk = sample_transaction(rng)
    attempt_number = 0
    hours_since_failure = 0.0
    hour_of_day = rng.randint(0, 23)
    total_reward = 0.0
    trace = []

    while True:
        state = make_state(failure_reason, amt_bucket, attempt_number, risk, hours_since_failure)
        valid = gr.valid_actions(attempt_number, hours_since_failure, amt_bucket, risk, hour_of_day)

        if training:
            action = policy.act_epsilon_greedy(state, valid, epsilon, rng)
        else:
            action = policy.act_greedy(state, valid)

        if action == "stop":
            trace.append({"attempt": attempt_number, "action": action, "success": False, "reward": 0.0})
            if training:
                policy.update(state, action, 0.0, None, None, done=True)
            break

        success, reward = env.step(failure_reason, action, attempt_number, risk, amount, rng)
        total_reward += reward
        trace.append({"attempt": attempt_number, "action": action, "success": success, "reward": reward})

        attempt_number += 1
        hours_since_failure += gr.next_cooldown_hours(action)
        hour_of_day = (hour_of_day + gr.next_cooldown_hours(action)) % 24
        done = success or attempt_number >= gr.MAX_ATTEMPTS or hours_since_failure >= gr.FORCED_STOP_HOURS

        next_state = None
        next_valid = None
        if not done:
            next_state = make_state(failure_reason, amt_bucket, attempt_number, risk, hours_since_failure)
            next_valid = gr.valid_actions(attempt_number, hours_since_failure, amt_bucket, risk, hour_of_day)

        if training:
            policy.update(state, action, reward, next_state, next_valid, done)

        if done:
            break

    return {
        "failure_reason": failure_reason, "amount": amount, "amount_bucket": amt_bucket,
        "customer_risk": risk, "recovered": any(t["success"] for t in trace),
        "net_reward": total_reward, "attempts": len(trace), "trace": trace,
    }
