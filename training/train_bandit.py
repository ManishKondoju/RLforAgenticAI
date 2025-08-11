import pandas as pd, numpy as np
from tqdm import tqdm
import yaml
from env.generate_data import make_stream
from env.calendar_sim import CalendarSim
from tools.context import featurize
from tools.tool_outcome import ARMS, arm_reward
from training.bandit import LinUCB

def run():
    cfg = yaml.safe_load(open("configs.yml"))
    np.random.seed(cfg["random_seed"])

    # data
    train = make_stream(cfg["stream"]["n_items_train"])
    val = make_stream(cfg["stream"]["n_items_val"])

    cal = CalendarSim()

    # bandit
    algo = LinUCB(n_arms=len(ARMS), d=cfg["bandit"]["context_dim"], alpha=cfg["bandit"]["alpha"])

    logs = []
    cum_reward = 0.0
    for i, row in tqdm(train.iterrows(), total=len(train), desc="Train"):
        x = featurize(row)
        # calendar proxy
        cal_q = cal.score_placement(row["deadline_hours"], row["est_duration_min"])
        arm = algo.select(x)
        r = arm_reward(ARMS[arm], x, cal_q)
        algo.update(arm, x, r)
        cum_reward += r
        logs.append({"step": i+1, "arm": ARMS[arm], "reward": r, "cum_reward": cum_reward})

    # quick validation pass (no updates)
    val_rewards = []
    for _, row in val.iterrows():
        x = featurize(row); cal_q = cal.score_placement(row["deadline_hours"], row["est_duration_min"])
        arm = algo.select(x)
        r = arm_reward(ARMS[arm], x, cal_q)
        val_rewards.append(r)

    df = pd.DataFrame(logs)
    df.to_csv("plots/bandit_training.csv", index=False)
    print(f"Train avg reward: {df['reward'].mean():.3f}, Val avg reward: {np.mean(val_rewards):.3f}")
    print("Wrote plots/bandit_training.csv")

if __name__ == "__main__":
    run()
