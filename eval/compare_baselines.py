import os, sys, numpy as np, yaml, pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.controller_env import ControllerEnv, ACTIONS
from agents.controller_dqn import DQNAgent
from training.bandit import LinUCB
from tools.tool_outcome import ARMS
import torch

def make_env(steps_per_episode: int, seed: int):
    return ControllerEnv(steps_per_episode=steps_per_episode, seed=seed)

def run_policy(policy_fn, steps_per_episode=200, episodes=20, seed0=1, bandit=None):
    """Recreate the env with different seeds to get stable averages."""
    rewards, resolved, violations = [], [], []
    for i in range(episodes):
        env = make_env(steps_per_episode, seed=seed0 + i)
        s = env.reset()
        ep_r = 0.0
        for _ in range(steps_per_episode):
            a = policy_fn(s)
            s, r, d, info = env.step(a, bandit=bandit)
            ep_r += r
            if d:
                break
        rewards.append(ep_r)
        resolved.append(info.get("resolved", 0))
        violations.append(info.get("deadline_violations", 0.0))
    return float(np.mean(rewards)), float(np.mean(resolved)), float(np.mean(violations))

def greedy_baseline_policy(s):
    # if not summarized -> summarize; else classify; then schedule
    summarized = s[-2] > 0.5
    classified = s[-1] > 0.5
    if not summarized:
        return ACTIONS.index("summarize")
    if not classified:
        return ACTIONS.index("classify")
    return ACTIONS.index("schedule")

def dqn_policy(agent):
    def _p(s):
        with torch.no_grad():
            qs = agent.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(qs).item())
    return _p

def main():
    cfg = yaml.safe_load(open("configs.yml"))
    steps = int(cfg["dqn"]["steps_per_episode"])
    model_path = cfg["dqn"]["model_path"]

    # ----- Baseline -----
    base = run_policy(greedy_baseline_policy, steps_per_episode=steps, episodes=20, seed0=100, bandit=None)
    print(f"Baseline -> AvgReward:{base[0]:.3f}  Resolved:{base[1]:.1f}  Violations:{base[2]:.2f}")

    # ----- DQN (load trained) -----
    env_tmp = make_env(steps_per_episode=steps, seed=999)
    agent = DQNAgent(env_tmp.state_dim, env_tmp.n_actions, cfg["dqn"])
    if os.path.isfile(model_path):
        agent.q.load_state_dict(torch.load(model_path, map_location="cpu"))
        agent.tgt.load_state_dict(agent.q.state_dict())
        print(f"Loaded trained DQN from {model_path}")
    else:
        print("WARNING: No trained model found; results will be random.")

    dqn_no_bandit = run_policy(dqn_policy(agent), steps_per_episode=steps, episodes=20, seed0=200, bandit=None)
    print(f"DQN -> AvgReward:{dqn_no_bandit[0]:.3f}  Resolved:{dqn_no_bandit[1]:.1f}  Violations:{dqn_no_bandit[2]:.2f}")

    # ----- DQN + Bandit -----
    bandit = LinUCB(n_arms=len(ARMS), d=int(cfg["bandit"]["context_dim"]), alpha=float(cfg["bandit"]["alpha"]))
    dqn_with_bandit = run_policy(dqn_policy(agent), steps_per_episode=steps, episodes=20, seed0=300, bandit=bandit)
    print(f"DQN+Bandit -> AvgReward:{dqn_with_bandit[0]:.3f}  Resolved:{dqn_with_bandit[1]:.1f}  Violations:{dqn_with_bandit[2]:.2f}")

    # ----- Save CSV -----
    rows = [
        {"policy":"Baseline","avg_reward":base[0],"resolved":base[1],"violations":base[2]},
        {"policy":"DQN","avg_reward":dqn_no_bandit[0],"resolved":dqn_no_bandit[1],"violations":dqn_no_bandit[2]},
        {"policy":"DQN+Bandit","avg_reward":dqn_with_bandit[0],"resolved":dqn_with_bandit[1],"violations":dqn_with_bandit[2]},
    ]
    os.makedirs("plots", exist_ok=True)
    out_path = "plots/baseline_vs_dqn.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
