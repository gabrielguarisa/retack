from sklearn import datasets
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from retack.experiment import Experiment
from retack.optimization import OptunaOptimizer

ds = datasets.load_boston()

em = Experiment(
    models=[DecisionTreeRegressor], metric_funcs=[mean_absolute_error]
)

results = em.run(ds.data, ds.target)

print(results)

em.set_optimizer(
    "DecisionTreeRegressor",
    OptunaOptimizer,
    metric_func=mean_absolute_error,
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
)

best_params, best_metric, model = em.optimize(
    "DecisionTreeRegressor",
    ds.data,
    ds.target,
    return_model=True,
    n_trials=100,
)

print(best_params, best_metric, model)
