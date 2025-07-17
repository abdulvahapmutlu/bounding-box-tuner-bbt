import random
from bbt.tuner import adaptive_top2_box_tuner
from bbt.utils import param_space_dict

def dummy_objective(params, train_ds, val_ds, epoch):
    """
    A toy objective: score = sum(params) + small noise + slight epoch boost
    """
    base = sum(params.values())
    noise = random.uniform(-0.1, 0.1)
    return base + 0.01 * epoch + noise

if __name__ == "__main__":
    best_params, best_score, trials_log, elapsed = adaptive_top2_box_tuner(
        train_dataset=None,
        val_dataset=None,
        param_space_dict=param_space_dict,
        objective_fn=dummy_objective,
        max_trials=20,
        init_samples=5,
        early_stopping_rounds=5,
        max_epochs=5,
    )
    print("Best params:", best_params)
    print("Best score :", best_score)
    print(f"Ran {len(trials_log)} trials in {elapsed:.2f}s")
e
