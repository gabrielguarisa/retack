from sklearn import datasets
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

import retack as rk

ds = datasets.load_boston()

oo = rk.OptunaOptimizer(
    model=DecisionTreeRegressor,
    model_args={
        "criterion": {
            "type": "suggest_categorical",
            "args": {"choices": ["mse", "friedman_mse", "mae", "poisson"]},
        },
        "max_depth": {
            "type": "suggest_int",
            "args": {"low": 4, "high": 10},
        },
    },
    metric_func=mean_absolute_error,
    direction="minimize",
)

results = oo.run(ds.data, ds.target, n_trials=100)

print(results)
