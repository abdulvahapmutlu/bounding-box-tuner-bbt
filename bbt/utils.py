import random

# Example parameter-space dictionary.
# Adapt keys, bounds, and types to your own models.
param_space_dict = {
    'learning_rate': {'type': 'float', 'bounds': (1e-5, 1e-1)},
    'batch_size':    {'type': 'int',   'bounds': (16, 128)},
    # add more hyperparameters here...
}

def sample_valid_params(param_space):
    """
    Sample a random set of hyperparameters from the full space.
    - param_space: dict of {name: {'type':'int'/'float', 'bounds':(lo, hi)}}
    """
    params = {}
    for name, info in param_space.items():
        lo, hi = info['bounds']
        if info['type'] == 'int':
            params[name] = random.randint(lo, hi)
        elif info['type'] == 'float':
            params[name] = random.uniform(lo, hi)
        else:
            raise ValueError(f"Unknown type for {name}: {info['type']}")
    return params

def sample_valid_params_bounding_box(box):
    """
    Sample uniformly within a given bounding box.
    - box: dict of {name: (lo, hi)}
    """
    params = {}
    for name, (lo, hi) in box.items():
        # infer int vs float by input types
        if isinstance(lo, int) and isinstance(hi, int):
            params[name] = random.randint(lo, hi)
        else:
            params[name] = random.uniform(lo, hi)
    return params
