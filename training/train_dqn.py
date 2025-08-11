import os, sys, yaml, numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.controller_env import ControllerEnv
from training.bandit import LinUCB
from tools.tool_outcome import ARMS
from agents.controller_dqn import DQNAgent
import pandas as pd
from tqdm import trange
import torch

def main():
    cfg = yaml.safe_load(open("configs.yml"))
    np.random.seed(cfg["random_seed"])

    # bandit for tool selection inside the env loop
    bandit = LinUCB(n_arms=len(ARMS), d=cfg["bandit"]["context_dim"], alpha=cfg["bandit"]["alpha"])

    env = ControllerEnv(steps_per_episode=int(cfg["dqn"]["steps_per_episode"]), seed=cfg["random_seed"])
    agent = DQNAgent(state_dim=env.state_dim, n_actions=env.n_actions, cfg=cfg["dqn"])

    os.makedirs("models", exist_ok=True)
    model_path = cfg["dqn"]["model_path"]

    # Optional: resume if model exists
    if bool(cfg["dqn"].get("resume_if_exists", False)) and os.path.isfile(model_path):
        agent.q.load_state_dict(torch.load(model_path, map_location="cpu"))
        agent.tgt.load_state_dict(agent.q.state_dict())
        print(f"Resumed weights from {model_path}")

    logs = []
    for ep in trange(int(cfg["dqn"]["episodes"]), desc="DQN"):
        s = env.reset()
        ep_reward, ep_loss = 0.0, []
        for t in range(int(cfg["dqn"]["steps_per_episode"])):
            a = agent.act(s)
            s2, r, done, info = env.step(a, bandit=bandit)
            agent.push(s, a, r, s2, float(done))
            loss = agent.train_step()
            if loss is not None: ep_loss.append(loss)
            ep_reward += r
            s = s2
            if done: break
        logs.append({
            "episode": ep+1,
            "reward": ep_reward,
            "loss": np.mean(ep_loss) if ep_loss else np.nan,
            "resolved": info.get("resolved", 0),
            "deadline_violations": info.get("deadline_violations", 0),
            "epsilon": agent.eps
        })

        # Save checkpoint every 25 episodes
        if (ep+1) % 25 == 0:
            torch.save(agent.q.state_dict(), model_path)

    # Final save
    torch.save(agent.q.state_dict(), model_path)

    df = pd.DataFrame(logs)
    os.makedirs("plots", exist_ok=True)
    df.to_csv("plots/dqn_training.csv", index=False)
    print("Saved plots/dqn_training.csv")
    print("Last 10 ep avg reward:", df["reward"].tail(10).mean())
    print("Deadline violations (↓ better), last 10 avg:", df["deadline_violations"].tail(10).mean())
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()
