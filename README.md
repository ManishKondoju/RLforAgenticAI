# RL for Agentic AI – Smart Triage RL System

> A hybrid reinforcement learning system for triage & scheduling that combines **DQN (controller)**, **Contextual Bandit (tool selection)**, and **PPO (scheduler)**. This repo includes training scripts, evaluation, plots, and a Streamlit dashboard.

---

## 🧭 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quickstart (Chronological Runbook)](#quickstart-chronological-runbook)
- [Configuration](#configuration)
- [Results](#results)
- [Visualization Dashboard](#visualization-dashboard)
- [Troubleshooting](#troubleshooting)
- [Ethics](#ethics)
- [License](#license)

---

## Overview
The **Smart Triage RL System** learns how to prioritize, process, and schedule incoming tasks. It uses:
- **DQN** to choose high-level actions (e.g., summarize, classify, schedule).
- **Contextual Bandit (LinUCB)** to select the best tool when summarization is requested.
- **PPO** to propose time slots; its output is used as a **quality shaping signal** for the controller.

The hybrid approach achieves **higher reward**, **max task resolution**, and **minimal violations** compared to a baseline.

---

## Architecture
<img width="468" height="175" alt="image" src="https://github.com/user-attachments/assets/52726e47-a0c5-45ba-a500-b76127815e78" />


![System Architecture](docs/architecture.png)

**Flow (runtime):**
1. Incoming tasks → **ControllerEnv** builds state.
2. **DQN** selects the next high-level action.
3. If `summarize` → **LinUCB Bandit** picks a tool and receives immediate reward.
4. If `schedule` → **PPO Scheduler** proposes a start slot; wrapper returns **quality** (lateness computed in ControllerEnv).
5. ControllerEnv updates metrics: **avg reward**, **resolved**, **violations**.
6. Evaluation writes CSVs → **plots** and **dashboard** consume them.

---

## Project Structure
```
smart-triage-rl/
│── agents/
│   ├── controller_dqn.py         # DQN agent
│   ├── bandit_linucb.py          # LinUCB bandit policy
│   └── scheduler_policy.py       # PPO wrapper (load SB3 model)
│── env/
│   └── controller_env.py         # Custom environment (state, step, reward)
│── training/
│   ├── train_bandit.py           # Train contextual bandit
│   ├── train_dqn.py              # Train DQN controller
│   └── train_ppo.py              # Train PPO scheduler (SB3)
│── eval/
│   ├── compare_baselines.py      # (optional) DQN vs Baseline
│   ├── compare_all.py            # Baseline, DQN, DQN+Bandit, Full
│   ├── plot_compare.py           # Bar charts (avg reward, resolved, violations)
│   └── stats.py                  # Aggregate across seeds + t-tests
│── dash/
│   └── app.py                    # Streamlit dashboard
│── models/                       # Saved models (created automatically)
│── plots/                        # CSVs and plots (created automatically)
│── configs.yml                   # All hyperparameters (edit here)
│── requirements.txt
│── README.md
└── docs/
    └── architecture.png          # <— put your diagram here
```

---

## Prerequisites
- **Python**: 3.10–3.11 recommended  
- **Virtual env** (recommended): `python -m venv venv && source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows)

Install dependencies:
```bash
pip install -r requirements.txt
# If you didn't install SB3 extras yet:
pip install "stable-baselines3[extra]" tqdm rich
```

---

## Quickstart (Chronological Runbook)
> From the repo root (`smart-triage-rl/`). This is the exact order we recommend.

### 0) Optional: Clean folders
```bash
mkdir -p models plots docs
```

### 1) Train Contextual Bandit (tool selection)
```bash
python -m training.train_bandit
# Output: plots/bandit_training.csv
```

### 2) Train DQN Controller
```bash
python -m training.train_dqn
# Outputs:
#   models/controller_dqn.pt
#   plots/dqn_training.csv
```

### 3) Train PPO Scheduler (SB3)
```bash
python -m training.train_ppo
# Output: models/scheduler_ppo.zip
```

### 4) Evaluate All Variants
```bash
python -m eval.compare_all
# Output: plots/compare_all.csv
```

### 5) Plot Comparison Charts
```bash
python -m eval.plot_compare
# Outputs (example):
#   plots/avg_reward_plot.png
#   plots/resolved_plot.png
#   plots/violations_plot.png
```

### 6) (Optional) Multi-Seed Stats + t-tests
```bash
pip install scipy
python -m eval.stats
# Output: plots/stats_summary.csv + printed t-tests
```

### 7) Launch the Dashboard
```bash
streamlit run dash/app.py
```

---

## Configuration
Edit hyperparameters and paths in **`configs.yml`**. Example:
```yaml
seed: 123
env:
  steps_per_episode: 200
  lateness_penalty: 1.0
  resolve_reward: 1.0
bandit:
  alpha: 0.6       # exploration for LinUCB
dqn:
  lr: 0.001        # learning rate (float!)
  gamma: 0.99
  buffer_size: 50000
  batch_size: 64
  target_update: 250
  episodes: 300
ppo:
  total_timesteps: 150000
  n_steps: 1024
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
```

---

## Results
Representative results across 5 seeds × 20 episodes:
| Variant                | Avg Reward | Resolved | Violations |
|------------------------|-----------:|---------:|-----------:|
| Baseline               | ~157       | 66.00    | 0.59       |
| DQN                    | ~346       | 199.25   | 0.04       |
| DQN + Bandit           | ~347       | 199.99   | 0.09       |
| DQN + Bandit + PPO     | **~360**   | **200.00** | 0.09     |

- **Full (DQN+Bandit+PPO)** gives the highest reward and max resolutions with low violations.  
- See `plots/compare_all.csv` and generated PNGs for your exact run.

---

## Visualization Dashboard
Run:
```bash
streamlit run dash/app.py
```
Tabs include:
- **Variant Comparison** (bar charts)
- **Training Curves** (DQN reward vs episode)
- **Bandit Analytics** (cumulative reward + arm counts)
- **Episode Trace** (step-by-step decisions, rewards)

---

## Troubleshooting
- `ModuleNotFoundError: No module named 'env'`  
  → Run from repo root (`smart-triage-rl/`), or add root to `PYTHONPATH`:
  ```bash
  export PYTHONPATH=$(pwd)
  ```
- YAML error (`while parsing a block mapping`)  
  → Check indentation in `configs.yml` and ensure numeric types are **not quoted**.
- PPO progress bar crash (`You must install tqdm and rich`)  
  → `pip install "stable-baselines3[extra]" tqdm rich`
- `SettingWithCopyWarning` in pandas  
  → Use `.copy()` before mutation, or `.at[]` for single-cell updates.

---

## Ethics
We include fairness-aware prioritization, transparent decision logs, privacy-safe data handling, and human-in-the-loop override for high-risk decisions.

---

## License
MIT License. See `LICENSE` for details.
