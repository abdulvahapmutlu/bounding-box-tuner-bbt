import time
import random
import statistics
from .utils import sample_valid_params, sample_valid_params_bounding_box

def adaptive_top2_box_tuner(
    train_dataset,
    val_dataset,
    param_space_dict,
    objective_fn,
    max_trials=50,
    init_samples=10,
    early_stopping_rounds=30,
    max_epochs=5,
):
    """
    Adaptive Top-2 Bounding-Box Tuner with:
      • Median-based partial-training pruning
      • Adaptive exploration → exploitation schedule

    Args:
      train_dataset, val_dataset: your data or DataLoader handles
      param_space_dict: see bbt/utils.py
      objective_fn(params, train_ds, val_ds, epoch) -> float
      max_trials: total candidate limit
      init_samples: random warm-up trials
      early_stopping_rounds: global stop if no improvement
      max_epochs: max epochs per trial

    Returns:
      best_params (dict), best_score (float),
      trials (list of dict), total_time (float)
    """
    trials = []
    # keep track of past epoch scores for median pruning
    rung_results = {e: [] for e in range(1, max_epochs + 1)}

    def evaluate_with_median_pruning(params):
        epoch_scores = {}
        for e in range(1, max_epochs + 1):
            score = objective_fn(params, train_dataset, val_dataset, epoch=e)
            epoch_scores[e] = score

            if len(rung_results[e]) >= 1:
                med = statistics.median(rung_results[e])
                if score < med:
                    return 0.0, True, epoch_scores  # pruned

            rung_results[e].append(score)

        return epoch_scores[max_epochs], False, epoch_scores

    start_time = time.time()

    # 1) Warm-up
    for _ in range(init_samples):
        cand = sample_valid_params(param_space_dict)
        score, pruned, epoch_scores = evaluate_with_median_pruning(cand)
        trials.append({
            "params": cand,
            "score": score,
            "pruned": pruned,
            "epoch_scores": epoch_scores
        })

    trials.sort(key=lambda t: t["score"], reverse=True)
    best_params = trials[0]["params"]
    best_score  = trials[0]["score"]
    no_improve  = 0
    trials_done = len(trials)

    # 2) Iterative search
    while trials_done < max_trials:
        trials.sort(key=lambda t: t["score"], reverse=True)
        P1 = trials[0]["params"]
        P2 = trials[1]["params"] if len(trials) > 1 else P1

        # build bounding box around top-2
        box = {k: (min(P1[k], P2[k]), max(P1[k], P2[k])) for k in P1}

        explore_prob = 0.35 if trials_done < max_trials / 2 else 0.10
        if random.random() < explore_prob:
            cand = sample_valid_params(param_space_dict)
        else:
            cand = sample_valid_params_bounding_box(box)

        score, pruned, epoch_scores = evaluate_with_median_pruning(cand)
        trials.append({
            "params": cand,
            "score": score,
            "pruned": pruned,
            "epoch_scores": epoch_scores
        })
        trials_done += 1

        if score > best_score:
            best_score  = score
            best_params = cand
            no_improve  = 0
        else:
            no_improve += 1

        if no_improve >= early_stopping_rounds:
            print(f"[Adaptive-Tuner] stopping: no improvement in {no_improve} trials.")
            break

    total_time = time.time() - start_time
    trials.sort(key=lambda t: t["score"], reverse=True)
    return best_params, best_score, trials, total_time
