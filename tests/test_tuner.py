import pytest
from bbt.tuner import adaptive_top2_box_tuner

def test_return_types_and_limits():
    # simple 0/1 space
    param_space = {'x': {'type': 'int', 'bounds': (0, 1)}}
    def obj_fn(params, td, vd, epoch):
        # monotonic in x
        return float(params['x'])

    best_params, best_score, trials, elapsed = adaptive_top2_box_tuner(
        train_dataset=None,
        val_dataset=None,
        param_space_dict=param_space,
        objective_fn=obj_fn,
        max_trials=5,
        init_samples=2,
        early_stopping_rounds=3,
        max_epochs=3,
    )
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert isinstance(trials, list)
    assert isinstance(elapsed, float)
    assert len(trials) <= 5

def test_pruning_all_zero():
    # objective always returns zero → everything pruned
    param_space = {'x': {'type': 'int', 'bounds': (0, 1)}}
    def zero_obj(params, td, vd, epoch):
        return 0.0

    _, _, trials, _ = adaptive_top2_box_tuner(
        train_dataset=None,
        val_dataset=None,
        param_space_dict=param_space,
        objective_fn=zero_obj,
        max_trials=4,
        init_samples=2,
        early_stopping_rounds=1,
        max_epochs=2,
    )
    # all trials should show pruned=True and score==0.0
    for t in trials:
        assert t['pruned'] is True
        assert t['score'] == 0.0
