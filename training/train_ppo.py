import os, yaml
from env.scheduler_env import SchedulerEnv
from stable_baselines3 import PPO

def main():
    cfg = yaml.safe_load(open("configs.yml"))
    ppo_cfg = cfg["ppo"]
    env = SchedulerEnv(
        slots_per_day=int(ppo_cfg["slots_per_day"]),
        slot_minutes=int(ppo_cfg["slot_minutes"]),
        tasks_per_day=int(ppo_cfg["tasks_per_day"]),
        seed=int(cfg["random_seed"]),
    )
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=int(ppo_cfg["total_timesteps"]), progress_bar=True)
    os.makedirs("models", exist_ok=True)
    model.save(ppo_cfg["model_path"])
    print(f"Saved PPO scheduler to {ppo_cfg['model_path']}.zip")

if __name__ == "__main__":
    main()
