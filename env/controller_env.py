import numpy as np
from env.generate_data import make_stream
from env.calendar_sim import CalendarSim
from tools.context import featurize
from tools.tool_outcome import ARMS, arm_reward

ACTIONS = ["summarize", "classify", "request_info", "schedule", "defer", "discard", "escalate"]

class ControllerEnv:
    def __init__(self, steps_per_episode=200, seed=42):
        self.steps_per_episode = steps_per_episode
        self.rng = np.random.default_rng(seed)
        self.cal = CalendarSim()
        self.reset_buffers()

    def reset_buffers(self):
        self.stream = make_stream(self.steps_per_episode + 50)
        self.ptr = 0
        self.queue_len = 0
        self.done_steps = 0
        self.current_item = None
        self.resolved = 0
        self.deadline_violations = 0

    def _get_item(self):
        if self.ptr >= len(self.stream):
            self.ptr = 0
        row = self.stream.iloc[self.ptr]
        self.ptr += 1
        return row

    def _encode_state(self, row, partial_progress):
        x = featurize(row)
        return np.hstack([
            x,
            np.array([self.queue_len/50.0, partial_progress["summarized"], partial_progress["classified"]], dtype=float)
        ]).astype(np.float32)

    def reset(self):
        self.reset_buffers()
        self.current_item = self._get_item()
        self.partial = {"summarized":0.0, "classified":0.0}
        s = self._encode_state(self.current_item, self.partial)
        return s

    def step(self, action_idx: int, bandit=None, scheduler=None):
        action = ACTIONS[action_idx]
        r = -0.01
        done = False

        cal_q = self.cal.score_placement(self.current_item["deadline_hours"], self.current_item["est_duration_min"])

        if action == "summarize":
            if bandit is not None:
                x = featurize(self.current_item)
                arm = bandit.select(x)
                r_step = arm_reward(ARMS[arm], x, cal_q)
                bandit.update(arm, x, r_step)
                r += 0.2 * (r_step - 0.5)
            self.partial["summarized"] = 1.0

        elif action == "classify":
            # avoid SettingWithCopyWarning by copying the Series when we mutate elsewhere
            amb = float(self.current_item["ambiguity"])
            correct = 1.0 if (amb < 0.7 or self.partial["summarized"] > 0.5) else 0.0
            r += 0.3*correct - 0.05*amb
            self.partial["classified"] = 1.0

        elif action == "request_info":
            # mutate a copy of the current row to avoid chained assignment warning
            self.current_item = self.current_item.copy()
            self.current_item.at["ambiguity"] = max(0.0, float(self.current_item["ambiguity"]) - 0.25)
            r += 0.05

        elif action == "schedule":
            # Use PPO scheduler (if provided) ONLY for placement quality shaping
            if callable(scheduler):
                sched_quality, _ignored = scheduler(self.current_item)
                cal_q = float(np.clip(sched_quality, 0.0, 1.0))

            # Keep a single, consistent lateness rule here
            late_flag = (self.current_item["est_duration_min"] + 60) > self.current_item["deadline_hours"]*60

            if late_flag:
                r -= 1.0
                self.deadline_violations += 1
            else:
                # ↑ Give PPO quality more influence (0.7 vs 0.5)
                # was + 0.7*cal_q
                r += 1.0 + 0.2*self.partial["summarized"] + 0.2*self.partial["classified"] + 0.9*cal_q

            self.resolved += 1
            self.current_item = self._get_item()
            self.queue_len = max(0, self.queue_len - 1)
            self.partial = {"summarized":0.0, "classified":0.0}

        elif action == "defer":
            self.queue_len += 1
            r -= 0.05

        elif action == "discard":
            low = (self.current_item["sender_importance"] <= 2) and (self.current_item["deadline_hours"] > 48)
            r += 0.2 if low else -0.6
            self.current_item = self._get_item()
            self.partial = {"summarized":0.0, "classified":0.0}

        elif action == "escalate":
            tight = self.current_item["deadline_hours"] < 8
            r += 0.4 if tight else -0.1
            self.resolved += 1
            self.current_item = self._get_item()
            self.partial = {"summarized":0.0, "classified":0.0}

        self.done_steps += 1
        if self.done_steps >= self.steps_per_episode:
            done = True

        s_next = self._encode_state(self.current_item, self.partial)
        info = {"resolved": self.resolved, "deadline_violations": self.deadline_violations}
        return s_next, float(r), done, info

    @property
    def state_dim(self):
        return 13

    @property
    def n_actions(self):
        return len(ACTIONS)
