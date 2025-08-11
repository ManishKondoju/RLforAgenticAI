import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("s","a","r","s2","d"))

class QNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x): return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, n_actions, cfg):
        self.device = torch.device("cpu")
        self.q = QNet(state_dim, n_actions).to(self.device)
        self.tgt = QNet(state_dim, n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())

        # Cast everything to safe types (prevents YAML string issues)
        lr = float(cfg["lr"])
        self.opt = optim.Adam(self.q.parameters(), lr=lr)

        self.gamma = float(cfg["gamma"])
        self.batch = int(cfg["batch_size"])
        self.buffer = deque(maxlen=int(cfg["buffer_size"]))
        self.start_after = int(cfg["start_training_after"])
        self.target_every = int(cfg["target_update_every"])
        self.steps = 0
        self.n_actions = n_actions

        self.eps = float(cfg["eps_start"])
        self.eps_end = float(cfg["eps_end"])
        self.eps_decay = float(cfg["eps_decay"])

    def act(self, s):
        import random
        self.steps += 1
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            qs = self.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(qs).item())

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def train_step(self):
        if len(self.buffer) < max(self.batch, self.start_after):
            return None
        import random
        batch = random.sample(self.buffer, self.batch)
        b = Transition(*zip(*batch))
        s  = torch.tensor(np.stack(b.s), dtype=torch.float32)
        a  = torch.tensor(b.a, dtype=torch.int64).unsqueeze(1)
        r  = torch.tensor(b.r, dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(np.stack(b.s2), dtype=torch.float32)
        d  = torch.tensor(b.d, dtype=torch.float32).unsqueeze(1)

        qsa = self.q(s).gather(1, a)
        with torch.no_grad():
            max_next = self.tgt(s2).max(1, keepdim=True)[0]
            target = r + (1 - d) * self.gamma * max_next
        loss = nn.functional.smooth_l1_loss(qsa, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        if self.steps % self.target_every == 0:
            self.tgt.load_state_dict(self.q.state_dict())

        # decay epsilon
        self.eps = max(self.eps_end, self.eps * float(self.eps_decay))
        return float(loss.item())
