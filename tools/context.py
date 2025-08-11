import numpy as np

def featurize(row):
    x = np.array([
        row["text_len"]/1000.0,
        row["sender_importance"]/5.0,
        row["deadline_hours"]/120.0,
        row["est_duration_min"]/180.0,
        row["ambiguity"],
        row["school_topic"],
        row["work_topic"],
        row["noise"]/3.0,
        row["difficulty"],
        (row["deadline_hours"] < 8)*1.0
    ], dtype=float)
    return x
