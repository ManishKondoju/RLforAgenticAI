import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SchedulerEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, slots_per_day=18, slot_minutes=30, tasks_per_day=32, seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.slot_minutes = slot_minutes
        self.N = slots_per_day
        self.tasks_per_day = tasks_per_day
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4 + self.N,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.N)
        self.reset(seed=seed)

    def _sample_task(self):
        return dict(
            deadline_hours=int(self.rng.integers(2, 120)),
            est_duration_min=int(self.rng.integers(15, 180)),
            sender_importance=int(self.rng.integers(1, 6)),
            ambiguity=float(self.rng.random()),
        )

    def _obs(self):
        t = self.current_task
        feats = np.array([
            t["deadline_hours"]/120.0,
            t["est_duration_min"]/180.0,
            t["sender_importance"]/5.0,
            t["ambiguity"],
        ], dtype=np.float32)
        return np.concatenate([feats, self.occ.astype(np.float32)], axis=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.occ = np.zeros(self.N, dtype=np.int32)
        self.i = 0
        self.current_task = self._sample_task()
        return self._obs(), {}

    def step(self, action):
        done = False
        t = self.current_task
        start = int(action)
        dur_slots = max(1, int(np.ceil(t["est_duration_min"]/self.slot_minutes)))
        end = start + dur_slots

        overlap = np.any(self.occ[start:min(end, self.N)] == 1) if start < self.N else True
        after_hours = end > self.N
        deadline_slots = int(t["deadline_hours"]*60/self.slot_minutes)
        finish_slot = end
        late = finish_slot > min(deadline_slots, self.N + deadline_slots)

        reward = 0.0
        if overlap or after_hours:
            reward -= 0.7
        else:
            left = start
            while left-1 >= 0 and self.occ[left-1]==0: left -= 1
            right = start
            while right < self.N and self.occ[right]==0: right += 1
            block = right - left
            reward += 0.1*block
            reward += 0.8 if not late else -0.5
            if after_hours: reward -= 0.2
            self.occ[start:min(end, self.N)] = 1

        self.i += 1
        if self.i >= self.tasks_per_day:
            done = True
        self.current_task = self._sample_task()
        return self._obs(), float(np.clip(reward, -1.0, 1.0)), done, False, {}
