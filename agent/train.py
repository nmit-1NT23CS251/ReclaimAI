"""Train the Q-learning recovery policy and save it to agent/q_table.json."""
import random

from agent.policy import QLearningPolicy, run_episode, EPSILON_START, EPSILON_END, EPSILON_DECAY_EPISODES

EPISODES = 500000


def epsilon_at(ep):
    frac = min(1.0, ep / EPSILON_DECAY_EPISODES)
    return EPSILON_START + frac * (EPSILON_END - EPSILON_START)


def main():
    rng = random.Random(7)
    policy = QLearningPolicy()

    recovered = 0
    reward_sum = 0.0
    window = 2000

    for ep in range(1, EPISODES + 1):
        eps = epsilon_at(ep)
        result = run_episode(policy, rng, training=True, epsilon=eps)
        recovered += int(result["recovered"])
        reward_sum += result["net_reward"]

        if ep % window == 0:
            print(f"ep {ep:6d}  eps={eps:.3f}  recovery_rate={recovered/window:.3f}  "
                  f"avg_net_reward={reward_sum/window:8.1f}  states_seen={len(policy.q)}")
            recovered = 0
            reward_sum = 0.0

    policy.save("agent/q_table.json")
    print(f"Saved Q-table with {len(policy.q)} states to agent/q_table.json")


if __name__ == "__main__":
    main()
