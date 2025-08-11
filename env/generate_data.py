import numpy as np, pandas as pd
rng = np.random.default_rng(42)

def make_stream(n):
    # Synthetic "emails/tasks"
    # Features: length, sender_importance, deadline_hours, est_duration, ambiguity, domain flags...
    X = pd.DataFrame({
        "text_len": rng.integers(20, 1000, size=n),
        "sender_importance": rng.integers(1, 6, size=n),          # 1–5
        "deadline_hours": rng.integers(2, 120, size=n),           # within 5 days
        "est_duration_min": rng.integers(15, 180, size=n),
        "ambiguity": rng.random(n),
        "school_topic": rng.integers(0, 2, size=n),
        "work_topic": rng.integers(0, 2, size=n),
        "noise": rng.normal(0, 1, size=n)
    })
    # “True” difficulty drives tool effectiveness & schedule pressure
    X["difficulty"] = (X["ambiguity"]*0.6 + (X["text_len"]>600)*0.2 + (X["deadline_hours"]<12)*0.2).astype(float)
    return X

if __name__ == "__main__":
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    train = make_stream(8000); val = make_stream(1000); test = make_stream(1000)
    train.to_csv("data/train_stream.csv", index=False)
    val.to_csv("data/val_stream.csv", index=False)
    test.to_csv("data/test_stream.csv", index=False)
    print("Wrote data/*.csv")
