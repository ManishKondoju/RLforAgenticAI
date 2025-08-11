# RL for Agentic AI – Smart Triage RL System

## 📌 Project Overview
This project implements a **Reinforcement Learning (RL) system** for intelligent task triage and scheduling.  
It leverages **DQN, Contextual Bandits, and PPO** to optimize task resolution and minimize violations in a simulated environment.

The system is designed to simulate **agentic AI decision-making** in environments requiring priority-based task handling.

---

## 🚀 Features
- **Multi-Algorithm RL**: DQN, Contextual Bandits, and PPO integration.
- **Performance Tracking**: Compare baseline vs trained models.
- **Statistical Validation**: Paired t-tests to verify improvements.
- **Interactive Visualization**: Learning curves, comparative bar plots, and agent performance analysis.
- **Scalable Design**: Extendable to multi-agent setups and real-world scheduling.

---

## 📂 Project Structure
```
smart-triage-rl/
│── env/                   # Custom RL environment
│── agents/                # RL algorithms (DQN, PPO, Bandit)
│── eval/                  # Evaluation scripts and plots
│── models/                # Saved trained models
│── plots/                 # Generated performance plots
│── dash/                  # Streamlit dashboard for visualization
│── main.py                # Training + evaluation script
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```
---

## ⚙️ Installation
```bash
git clone https://github.com/ManishKondoju/RLforAgenticAI.git
cd RLforAgenticAI
pip install -r requirements.txt
```

---

## ▶️ Usage

### **1. Train & Evaluate Models**
```bash
python main.py
```

### **2. Launch Visualization Dashboard**
```bash
streamlit run dash/app.py
```

---

## 📊 Experimental Methodology
1. **Training Phase**: Train each agent (Baseline, DQN, DQN+Bandit, DQN+Bandit+PPO).
2. **Evaluation Phase**: Run episodes, collect rewards, resolutions, and violations.
3. **Analysis Phase**: Statistical tests and visualizations.

---

## 📈 Results Summary

| Variant              | Avg Reward | Resolved Tasks | Violations |
|----------------------|------------|---------------|------------|
| **Baseline**         | ~157       | 66.00          | 0.59       |
| **DQN**              | ~346       | 199.25         | 0.04       |
| **DQN + Bandit**     | ~347       | 199.99         | 0.09       |
| **DQN + Bandit + PPO** | ~360     | 200.00         | 0.09       |

✅ **DQN+Bandit+PPO** showed the highest improvement in reward and resolution rate.

---

## 🧠 RL Techniques Used
- **DQN** for value-based learning.
- **Contextual Bandit** for adaptive task selection.
- **PPO** for policy-based scheduling optimization.

---

## 🔮 Future Improvements
- Multi-agent collaboration.
- Human-in-the-loop feedback.
- More contextual features in Bandit models.
- Real-world deployment.

---

## ⚖ Ethical Considerations
- Fair task allocation.
- Transparency and explainability.
- Data privacy compliance.
- Avoidance of bias.

---

## 📜 License
This project is licensed under the MIT License.
