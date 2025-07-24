# Bounding-Box Tuner (BBT)

**Adaptive Top-2 Bounding-Box Hyperparameter Tuner**  
Lightweight hyperparameter search with median-based early pruning.

Bounding-Box Tuner (BBT) implements a simple but effective strategy to balance exploration and exploitation:

1. **Top-2 bounding box sampling**: At each iteration, define a hyper-rectangular “box” around your two best configurations and sample new candidates inside it.  
2. **Adaptive exploration schedule**: Start with 35 % global random sampling, then decay to 10 % to focus on the promising region.  (It can be adjusted)
3. **Median-based partial-training pruning**: After every epoch _e_, prune any trial whose validation score falls below the median of all previous epoch-_e_ scores.

---

## 🚀 Features

- **Easy integration**: Plug BBT into any training loop via a one-line objective function  
- **Fast partial training**: Each trial runs for up to `max_epochs` epochs, but can be cut short if it underperforms  
- **Minimal dependencies**: Only standard Python libraries (`numpy`, `scipy`, `statistics`)  
- **Configurable**: Control total trials (`max_trials`), warm-up samples (`init_samples`), early stops, and more

---

## 🔧 Installation

> **Note**: this is a “start small” release—no PyPI package or CI badges yet.  
> Install directly from GitHub:

```
git clone https://github.com/abdulvahapmutlu/bounding-box-tuner-bbt.git
cd bounding-box-tuner-bbt
pip install -r requirements.txt
````

---

## ⚡ Quickstart

```
from bbt.tuner import adaptive_top2_box_tuner
from bbt.utils import param_space_dict

def objective_fn(params, train_ds, val_ds, epoch):
    """
    Train one epoch with `params` on train_ds,
    evaluate on val_ds, and return a scalar metric.
    """
    # YOUR TRAINING LOOP HERE
    return val_accuracy

best_params, best_score, trials_log, elapsed = adaptive_top2_box_tuner(
    train_dataset=my_train_ds,
    val_dataset=my_val_ds,
    param_space_dict=param_space_dict,
    objective_fn=objective_fn,
    max_trials=30,
    init_samples=5,
    early_stopping_rounds=10,
    max_epochs=5,
    explore_rate_start=0.35,    # initial random-sampling probability
    explore_rate_end=0.10,      # final random-sampling probability
    n_workers=0                 # number of parallel workers
)

print("Best params:", best_params)
print("Best validation score:", best_score)
print(f"Ran {len(trials_log)} trials in {elapsed:.1f}s.")

```

---

## 📚 API Reference

### `adaptive_top2_box_tuner(...)`

```
adaptive_top2_box_tuner(
    train_dataset,
    val_dataset,
    param_space_dict: dict,
    objective_fn: Callable[[dict, Any, Any, int], float],
    max_trials: int = 50,
    init_samples: int = 10,
    early_stopping_rounds: int = 30,
    max_epochs: int = 5,
    explore_rate_start: float = 0.35,
    explore_rate_end: float = 0.10,
    n_workers: int = 1,
) -> Tuple[dict, float, List[dict], float]
```

* **Returns**

  1. `best_params` (dict): highest-scoring hyperparameter set
  2. `best_score` (float): corresponding validation metric
  3. `trials_log` (list of dict)
  4. `total_time` (float)

     * `"params"`: dict
     * `"score"`: float
     * `"pruned"`: bool
     * `"epoch_scores"`: `{epoch: score}`
  5. `total_time` (float): elapsed seconds

* **Key args**

  * `param_space_dict`: map each hyperparameter to its sampling info (min/max, type, etc.)
  * `objective_fn`: called once per epoch—return validation score
  * `max_trials`, `init_samples`, `early_stopping_rounds`, `max_epochs`: control search budget & pruning
  * `explore_rate_start`: starting probability of global random sampling
  * `explore_rate_end`: ending probability of global random sampling
  * `n_workers`: number of parallel processes (uses multiprocessing.Pool)

See [`bbt/utils.py`](bbt/utils.py) for helpers:

* `sample_valid_params(param_space_dict)`
* `sample_valid_params_bounding_box(box)`

---


## 🙌 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/YourFeature`
3. Write code & tests
4. Open a Pull Request

Please follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and include tests for new functionality.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

