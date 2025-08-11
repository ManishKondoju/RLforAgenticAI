import numpy as np

ARMS = ["RuleSummarizer","LLMSummarizer","HeuristicPriority","MLPriority","Skip"]

def arm_reward(arm: str, x: np.ndarray, calendar_quality: float) -> float:
    # x dims (see featurize): we craft biases so some arms do better on certain contexts
    text_len, sender_imp, deadline_norm, dur_norm, ambig, school, work, noise, diff, super_urgent = x

    base = 0.3 + 0.2*sender_imp + 0.2*calendar_quality - 0.1*diff
    if arm == "RuleSummarizer":
        # fast but struggles with high ambiguity/long text
        r = base - 0.15*ambig - 0.1*(text_len>0.6)
    elif arm == "LLMSummarizer":
        # slower but handles ambiguity/long text better
        r = base + 0.15*ambig + 0.1*(text_len>0.6) - 0.05  # slight latency penalty
    elif arm == "HeuristicPriority":
        # ok generally, worse when super urgent + long duration
        r = base - 0.1*(super_urgent) - 0.05*(dur_norm>0.5)
    elif arm == "MLPriority":
        # shines on work topics and medium ambiguity
        r = base + 0.1*work + 0.05*(0.2 < ambig < 0.7)
    else:  # Skip
        r = 0.05  # almost always bad
    # noise
    r += np.random.normal(0, 0.03)
    return float(np.clip(r, 0.0, 1.0))
