import numpy as np
from stable_baselines3 import PPO
from env.scheduler_env import SchedulerEnv

class PPOScheduler:
    """
    scheduler(item_row) -> (quality in [0,1], False)
    - Uses PPO to suggest a start slot
    - Snaps forward to the next feasible free block if needed
    - Maintains internal "day" occupancy and auto-resets when near full
    """
    def __init__(self, model_path: str, slots_per_day=18, slot_minutes=30, tasks_per_day=256, seed=123):
        self.env = SchedulerEnv(
            slots_per_day=slots_per_day,
            slot_minutes=slot_minutes,
            tasks_per_day=tasks_per_day,
            seed=seed
        )
        if model_path.endswith(".zip"):
            self.model = PPO.load(model_path)
        else:
            self.model = PPO.load(model_path + ".zip")
        self.placed = 0

    def reset_day(self):
        self.env.reset()
        self.placed = 0

    def _find_next_feasible_start(self, start, dur_slots):
        """Return earliest start >= start that fits without overlap/after-hours, else None."""
        N = self.env.N
        occ = self.env.occ
        for s in range(start, N):
            e = s + dur_slots
            if e > N:
                return None
            if np.all(occ[s:e] == 0):
                return s
        return None

    def __call__(self, item_row):
        deadline_hours = float(item_row["deadline_hours"])
        est_duration_min = float(item_row["est_duration_min"])
        sender_importance = float(item_row["sender_importance"])
        ambiguity = float(item_row["ambiguity"])

        feats = np.array([
            deadline_hours/120.0,
            est_duration_min/180.0,
            sender_importance/5.0,
            ambiguity,
        ], dtype=np.float32)
        obs = np.concatenate([feats, self.env.occ.astype(np.float32)], axis=0)

        # PPO suggestion
        action, _ = self.model.predict(obs, deterministic=True)
        start = int(action)
        dur_slots = max(1, int(np.ceil(est_duration_min / self.env.slot_minutes)))

        # Snap forward to a feasible slot
        snapped = self._find_next_feasible_start(start, dur_slots)
        if snapped is None:
            # If no space left today, reset day and try from slot 0
            self.reset_day()
            snapped = self._find_next_feasible_start(0, dur_slots)
            if snapped is None:
                # If still impossible (huge task), give zero quality
                return 0.0, False

        s, e = snapped, snapped + dur_slots
        self.env.occ[s:e] = 1
        self.placed += 1

        # Auto-reset when day > 90% full to avoid degeneracy
        if (self.env.occ.sum() > 0.9 * self.env.N) or (self.placed > 0.9 * self.env.N):
            self.reset_day()

        # Quality: favor larger focus blocks around placement
        left = s
        while left-1 >= 0 and self.env.occ[left-1] == 1:
            left -= 1
        right = s
        while right < self.env.N and self.env.occ[right] == 1:
            right += 1
        block = right - left

        # Late relative to deadline? (just reduces quality; lateness penalty handled in controller)
        deadline_slots = int(deadline_hours * 60 / self.env.slot_minutes)
        late = (e > min(deadline_slots, self.env.N + deadline_slots))

        quality = 0.55 + 0.03 * block  # a bit higher baseline so PPO signal matters
        if not late:
            quality += 0.25

        return float(np.clip(quality, 0.0, 1.0)), False  # controller handles lateness
