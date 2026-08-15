import numpy as np


def no_skill_baselines(y_true) -> dict[str, float]:
    """Reference baseline scores for a no-skill model"""
    p = float(np.mean(y_true))
    return {
        "prevalence": p,
        "accuracy": max(p, 1.0 - p),
        "roc_auc": 0.5,
        "pr_auc": p,
        "brier": p * (1.0 - p),
        "log_loss": float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))),
    }