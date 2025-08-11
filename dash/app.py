# dash/app.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import yaml
import torch
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from env.controller_env import ControllerEnv, ACTIONS
from agents.controller_dqn import DQNAgent
from training.bandit import LinUCB
from tools.tool_outcome import ARMS

# Optional: PPO scheduler wrapper (load lazily)
try:
    from agents.scheduler_policy import PPOScheduler
    HAS_PPO = True
except Exception:
    HAS_PPO = False

st.set_page_config(page_title="Smart Triage RL Dashboard", layout="wide")

# ---------- Load config ----------
cfg = yaml.safe_load(open("configs.yml"))
steps = int(cfg["dqn"]["steps_per_episode"])

# ---------- Helpers ----------
@st.cache_resource
def load_dqn(model_path: str):
    env = ControllerEnv(steps_per_episode=steps, seed=999)
    agent = DQNAgent(env.state_dim, env.n_actions, cfg["dqn"])
    if os.path.isfile(model_path):
        agent.q.load_state_dict(torch.load(model_path, map_location="cpu"))
        agent.tgt.load_state_dict(agent.q.state_dict())
    return agent

def baseline_policy(s):
    # summarize -> classify -> schedule
    if s[-2] <= 0.5:
        return ACTIONS.index("summarize")
    if s[-1] <= 0.5:
        return ACTIONS.index("classify")
    return ACTIONS.index("schedule")

def dqn_policy_fn(agent):
    def _p(s):
        with torch.no_grad():
            qs = agent.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(qs).item())
    return _p

def run_episode_traced(policy_fn, bandit=None, scheduler=None, seed=123, steps=steps):
    env = ControllerEnv(steps_per_episode=steps, seed=seed)
    if scheduler:
        scheduler.reset_day()
    s = env.reset()
    total = 0.0
    log = []
    for t in range(steps):
        a = policy_fn(s)
        s, r, d, info = env.step(a, bandit=bandit, scheduler=scheduler)
        total += r
        log.append({
            "t": t,
            "action": ACTIONS[a],
            "reward": r,
            "cum_reward": total,
            "resolved": info["resolved"],
            "violations": info["deadline_violations"],
        })
        if d:
            break
    return total, info, pd.DataFrame(log)

def occupancy_heatmap(scheduler):
    occ = scheduler.env.occ
    fig = px.imshow(
        np.array([occ], dtype=float),
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(color="Busy"),
        title="PPO Day Occupancy (1 = busy)"
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    return fig

@st.cache_data
def load_csv(path):
    if os.path.isfile(path):
        return pd.read_csv(path)
    return None

# ---------- Title ----------
st.title("Smart Triage & Scheduling — RL Dashboard")

# Preload models
agent = load_dqn(cfg["dqn"]["model_path"])
bandit_template = LinUCB(n_arms=len(ARMS), d=int(cfg["bandit"]["context_dim"]), alpha=float(cfg["bandit"]["alpha"]))

scheduler = None
if HAS_PPO and os.path.isfile(cfg["ppo"]["model_path"] + ".zip"):
    scheduler = PPOScheduler(
        model_path=cfg["ppo"]["model_path"],
        slots_per_day=int(cfg["ppo"]["slots_per_day"]),
        slot_minutes=int(cfg["ppo"]["slot_minutes"]),
        tasks_per_day=int(cfg["ppo"]["tasks_per_day"]),
        seed=int(cfg["random_seed"]),
    )

# ---------- Tabs ----------
tab_compare, tab_trace, tab_bandit, tab_training = st.tabs(
    ["📊 Variant Comparison", "🎬 Episode Trace", "🎯 Bandit Analytics", "📈 Training Curves"]
)

# ====== Variant Comparison ======
with tab_compare:
    st.subheader("Final Comparison (Baseline vs DQN vs Bandit vs PPO)")
    df = load_csv("plots/compare_all.csv")
    if df is None:
        st.info("Run `python -m eval.compare_all` to generate plots/compare_all.csv")
    else:
        # Enforce a nice order if present
        order = ["Baseline", "DQN", "DQN+Bandit", "DQN+Bandit+PPO"]
        df["policy"] = pd.Categorical(df["policy"], categories=[p for p in order if p in df["policy"].unique()], ordered=True)
        df = df.sort_values("policy")

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.bar(df, x="policy", y="avg_reward", text="avg_reward", title="Average Reward")
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(df, x="policy", y="resolved", text="resolved", title="Tasks Resolved per Episode")
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = px.bar(df, x="policy", y="violations", text="violations", title="Deadline Violations per Episode")
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Data source: plots/compare_all.csv")
        st.dataframe(df, use_container_width=True)

# ====== Episode Trace ======
with tab_trace:
    st.subheader("Single Episode Trace (actions, reward, cumulative reward)")
    mode = st.selectbox("Policy", ["Baseline", "DQN", "DQN+Bandit", "DQN+Bandit+PPO"])
    seed_val = st.number_input("Seed", min_value=1, max_value=999, value=123, step=1)
    steps_val = st.slider("Max steps", min_value=50, max_value=400, value=steps, step=10)
    if st.button("Run Trace"):
        if mode == "Baseline":
            policy = baseline_policy
            bandit = None
            sch = None
        else:
            policy = dqn_policy_fn(agent)
            bandit = bandit_template if ("Bandit" in mode) else None
            sch = scheduler if ("PPO" in mode and scheduler is not None) else None

        total, info, df_trace = run_episode_traced(policy, bandit=bandit, scheduler=sch, seed=seed_val, steps=steps_val)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Reward", f"{total:.2f}")
        m2.metric("Resolved", f"{info['resolved']}")
        m3.metric("Violations", f"{info['deadline_violations']}")
        st.line_chart(df_trace.set_index("t")[["reward", "cum_reward"]])
        st.write("Action sequence (tail):")
        st.dataframe(df_trace.tail(20), use_container_width=True)
        if sch is not None:
            st.plotly_chart(occupancy_heatmap(sch), use_container_width=True)

# ====== Bandit Analytics ======
with tab_bandit:
    st.subheader("Contextual Bandit (LinUCB) — Training Signals")
    dfb = load_csv("plots/bandit_training.csv")
    if dfb is None:
        st.info("Run `python -m training.train_bandit` to generate plots/bandit_training.csv")
    else:
        dfb["avg"] = dfb["cum_reward"] / dfb["step"]
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(dfb.set_index("step")[["avg"]])
            st.caption("Cumulative average reward over steps")
        with c2:
            arm_counts = dfb["arm"].value_counts().rename_axis("arm").reset_index(name="count")
            fig = px.bar(arm_counts, x="arm", y="count", title="Arm Selection Counts")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dfb.tail(10), use_container_width=True)

# ====== Training Curves ======
with tab_training:
    st.subheader("Training Curves")
    dqd = load_csv("plots/dqn_training.csv")
    if dqd is None:
        st.info("Run `python -m training.train_dqn` to generate plots/dqn_training.csv")
    else:
        dqd["reward_ma10"] = dqd["reward"].rolling(10).mean()
        st.line_chart(dqd.set_index("episode")[["reward", "reward_ma10"]])
        st.caption("DQN: episode reward and 10-episode moving average.")
    st.caption("PPO trained via Stable-Baselines3. For PPO episode returns, enable SB3 Monitor callbacks if desired.")
