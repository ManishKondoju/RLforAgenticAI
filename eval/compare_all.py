import os, sys, yaml, numpy as np, pandas as pd, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.controller_env import ControllerEnv, ACTIONS
from agents.controller_dqn import DQNAgent
from training.bandit import LinUCB
from tools.tool_outcome import ARMS
from agents.scheduler_policy import PPOScheduler

def make_env(steps_per_episode, seed):
    return ControllerEnv(steps_per_episode=steps_per_episode, seed=seed)

def greedy_baseline_policy(s):
    summarized = s[-2] > 0.5
    classified = s[-1] > 0.5
    if not summarized: return ACTIONS.index("summarize")
    if not classified: return ACTIONS.index("classify")
    return ACTIONS.index("schedule")

def dqn_policy(agent):
    def _p(s):
        with torch.no_grad():
            qs = agent.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(qs).item())
    return _p

def run_policy(policy_fn, steps_per_episode, episodes, seed0, bandit=None, scheduler=None):
    rewards, resolved, violations = [], [], []
    for i in range(episodes):
        env = make_env(steps_per_episode, seed0+i)
        if scheduler is not None:
            scheduler.reset_day()
        s = env.reset()
        ep_r = 0.0
        for _ in range(steps_per_episode):
            a = policy_fn(s)
            s, r, d, info = env.step(a, bandit=bandit, scheduler=scheduler)
            ep_r += r
            if d: break
        rewards.append(ep_r)
        resolved.append(info.get("resolved", 0))
        violations.append(info.get("deadline_violations", 0.0))
    return float(np.mean(rewards)), float(np.mean(resolved)), float(np.mean(violations))

def main():
    cfg = yaml.safe_load(open("configs.yml"))
    steps = int(cfg["dqn"]["steps_per_episode"])

    # Load DQN
    tmp_env = make_env(steps, seed=999)
    agent = DQNAgent(tmp_env.state_dim, tmp_env.n_actions, cfg["dqn"])
    model_path = cfg["dqn"]["model_path"]
    if os.path.isfile(model_path):
        agent.q.load_state_dict(torch.load(model_path, map_location="cpu"))
        agent.tgt.load_state_dict(agent.q.state_dict())
        print(f"Loaded DQN from {model_path}")
    else:
        print("WARNING: no DQN model found; results will be random.")

    # Prepare Bandit
    bandit = LinUCB(n_arms=len(ARMS), d=int(cfg["bandit"]["context_dim"]), alpha=float(cfg["bandit"]["alpha"]))

    # Prepare PPO Scheduler
    ppo_cfg = cfg["ppo"]
    ppo_path = ppo_cfg["model_path"]
    if os.path.isfile(ppo_path + ".zip"):
        scheduler = PPOScheduler(
            model_path=ppo_path,
            slots_per_day=int(ppo_cfg["slots_per_day"]),
            slot_minutes=int(ppo_cfg["slot_minutes"]),
            tasks_per_day=int(ppo_cfg["tasks_per_day"]),
            seed=int(cfg["random_seed"]),
        )
        print(f"Loaded PPO scheduler from {ppo_path}.zip")
    else:
        scheduler = None
        print("WARNING: PPO scheduler not found; skipping combined variant.")

    rows = []
    def log_row(name, res): rows.append({"policy":name,"avg_reward":res[0],"resolved":res[1],"violations":res[2]})

    # Baseline
    base = run_policy(greedy_baseline_policy, steps, episodes=20, seed0=100, bandit=None, scheduler=None)
    print("Baseline:", base); log_row("Baseline", base)

    # DQN
    dqn_res = run_policy(dqn_policy(agent), steps, episodes=20, seed0=200, bandit=None, scheduler=None)
    print("DQN:", dqn_res); log_row("DQN", dqn_res)

    # DQN + Bandit
    dqn_bandit = run_policy(dqn_policy(agent), steps, episodes=20, seed0=300, bandit=bandit, scheduler=None)
    print("DQN+Bandit:", dqn_bandit); log_row("DQN+Bandit", dqn_bandit)

    # DQN + Bandit + PPO
    if scheduler is not None:
        dqn_bandit_ppo = run_policy(dqn_policy(agent), steps, episodes=20, seed0=400, bandit=bandit, scheduler=scheduler)
        print("DQN+Bandit+PPO:", dqn_bandit_ppo); log_row("DQN+Bandit+PPO", dqn_bandit_ppo)

    os.makedirs("plots", exist_ok=True)
    out = "plots/compare_all.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
