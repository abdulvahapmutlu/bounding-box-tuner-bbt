# File: bbt/tuner.py

import time
import random
import statistics
from multiprocessing import Pool
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
    explore_rate_start=0.35,
    explore_rate_end=0.10,
    n_workers=1,
):
    """
    Adaptive Top-2 Bounding-Box Tuner with:
      • Median-based partial-training pruning (score <= median is pruned)
      • Configurable exploration schedule
      • Optional parallel evaluation
      • Any final_score == 0.0 is considered pruned

    Returns: (best_params, best_score, trials_log, total_time)
    """
    trials = []
    # Keep history of epoch scores for median pruning
    rung_results = {e: [] for e in range(1, max_epochs + 1)}

    def evaluate_one(params):
        epoch_scores = {}
        for e in range(1, max_epochs + 1):
            score = objective_fn(params, train_dataset, val_dataset, epoch=e)
            epoch_scores[e] = score

            # median-based pruning: prune if score <= median of past runs at this epoch
            if len(rung_results[e]) >= 1:
                med = statistics.median(rung_results[e])
                if score <= med:
                    return params, 0.0, True, epoch_scores

            # otherwise record for future medians
            rung_results[e].append(score)

        # all epochs completed
        final_score = epoch_scores[max_epochs]
        pruned = final_score == 0.0  # also mark zero-final-score runs as pruned
        return params, final_score, pruned, epoch_scores

    start_time = time.time()

    # ─── 1) Warm-up Random Sampling ───────────────────────────────────────────────
    for _ in range(init_samples):
        cand = sample_valid_params(param_space_dict)
        _, score, pruned, ep_scores = evaluate_one(cand)
        trials.append({
            "params": cand,
            "score": score,
            "pruned": pruned,
            "epoch_scores": ep_scores
        })

    # Initialize best from warm-up
    trials.sort(key=lambda t: t["score"], reverse=True)
    best_params = trials[0]["params"]
    best_score  = trials[0]["score"]
    no_improve  = 0
    trials_done = len(trials)

    # ─── 2) Iterative Search ─────────────────────────────────────────────────────
    while trials_done < max_trials:
        # Identify current Top-2 for bounding box
        trials.sort(key=lambda t: t["score"], reverse=True)
        P1 = trials[0]["params"]
        P2 = trials[1]["params"] if len(trials) > 1 else P1

        # Build bounding box around Top-2
        box = {k: (min(P1[k], P2[k]), max(P1[k], P2[k])) for k in P1}

        # Determine exploration vs. exploitation rate
        prob_explore = explore_rate_start if trials_done < max_trials / 2 else explore_rate_end

        # Prepare a batch of new candidates
        batch_size = min(n_workers, max_trials - trials_done)
        cands = []
        for _ in range(batch_size):
            if random.random() < prob_explore:
                cands.append(sample_valid_params(param_space_dict))
            else:
                cands.append(sample_valid_params_bounding_box(box))

        # Evaluate either in parallel or serially
        if n_workers > 1:
            with Pool(batch_size) as pool:
                results = pool.map(evaluate_one, cands)
        else:
            results = [evaluate_one(c) for c in cands]

        # Collect results and update best/no_improve
        for params, score, pruned, ep_scores in results:
            trials.append({
                "params": params,
                "score": score,
                "pruned": pruned,
                "epoch_scores": ep_scores
            })
            trials_done += 1

            if score > best_score:
                best_score, best_params, no_improve = score, params, 0
            else:
                no_improve += 1

        # Global early stopping
        if no_improve >= early_stopping_rounds:
            print(f"[Adaptive-Tuner] stopping: no improvement in {no_improve} trials.")
            break

    # Finalize
    total_time = time.time() - start_time
    trials.sort(key=lambda t: t["score"], reverse=True)
    return best_params, best_score, trials, total_time
