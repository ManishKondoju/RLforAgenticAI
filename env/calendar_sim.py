import numpy as np
from datetime import timedelta

class CalendarSim:
    """Toy calendar quality proxy for reward shaping."""
    def __init__(self, work_start=9, work_end=18, slot_minutes=30):
        self.work_start = work_start
        self.work_end = work_end
        self.slot = slot_minutes

    def score_placement(self, deadline_hours, est_duration_min):
        # Reward high if we can fit before deadline within work hours.
        # This is a lightweight proxy used by the bandit to estimate downstream quality.
        buffer = 60  # desired buffer minutes
        within_deadline = (est_duration_min + buffer) <= deadline_hours*60
        # Penalize long tasks near deadlines
        tightness = max(0.0, 1.0 - (est_duration_min / max(30, deadline_hours*10)))
        base = 0.6*tightness + (0.4 if within_deadline else 0.0)
        return float(np.clip(base, 0, 1))
