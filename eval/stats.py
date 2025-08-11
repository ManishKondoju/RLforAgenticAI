# eval/stats.py
import os, sys, yaml, numpy as np, pandas as pd, torch
from scipy.stats import ttest_rel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.compare_all import main as run_once  # reuses your compare_all writer

SEEDS = [11, 22, 33, 44, 55]

def read_row(path, policy):
    df = pd.read_csv(path)
    return df[df["policy"] == policy].iloc[0][["avg_reward","resolved","violations"]].to_numpy(dtype=float)

def main():
    yaml.safe_load(open("configs.yml"))  # just to fail fast if missing
    rows = {p: [] for p in ["Baseline","DQN","DQN+Bandit","DQN+Bandit+PPO"]}
    for s in SEEDS:
        # re-run compare_all with a different seed by temporarily editing the env var
        os.environ["RL_GLOBAL_SEED"] = str(s)  # optional if you wire it; else compare_all already randomizes
        run_once()
        r = "plots/compare_all.csv"
        for p in list(rows.keys()):
            if not os.path.exists(r): raise FileNotFoundError(r)
            try:
                rows[p].append(read_row(r, p))
            except Exception:
                # may not have PPO row if model missing; skip politely
                pass
    # aggregate
    out = {}
    for p, arr in rows.items():
        if not arr: continue
        X = np.vstack(arr)  # shape [seeds, 3]
        mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
        out[p] = {"avg_reward_mean":mu[0],"avg_reward_std":sd[0],"resolved_mean":mu[1],"resolved_std":sd[1],"violations_mean":mu[2],"violations_std":sd[2]}
    df = pd.DataFrame.from_dict(out, orient="index")
    os.makedirs("plots", exist_ok=True)
    df.to_csv("plots/stats_summary.csv")
    print(df)

    # paired t-test: DQN vs DQN+Bandit+PPO on reward & violations
    if len(rows["DQN"]) == len(rows["DQN+Bandit+PPO"]) and len(rows["DQN"]) >= 2:
        dqn = np.vstack(rows["DQN"])
        full = np.vstack(rows["DQN+Bandit+PPO"])
        t_r = ttest_rel(full[:,0], dqn[:,0], alternative="greater")
        t_v = ttest_rel(dqn[:,2], full[:,2], alternative="greater")  # want fewer violations in full → reverse
        print("Paired t-test (reward, full > dqn):", t_r)
        print("Paired t-test (violations, dqn > full):", t_v)

if __name__ == "__main__":
    try:
        from scipy.stats import ttest_rel  # check dependency
    except ImportError:
        print("Install scipy first: pip install scipy")
        raise
    main()
