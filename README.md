# retack

The source code is currently hosted on GitHub at: https://github.com/gabrielguarisa/retack

## Installation

```shell
pip install retack
```

## Examples

### Simple experiment

```python
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, jaccard_score
from sklearn.svm import SVC

import retack as rk

ds = datasets.load_iris()

em = rk.Experiment(
    models=[SVC(), RandomForestClassifier(), AdaBoostClassifier()],
    metric_funcs={"accuracy": accuracy_score, "jaccard": jaccard_score},
    metric_funcs_args={"jaccard": {"average": "micro"}},
)

results = em.run(ds.data, ds.target)
```

### Using optimizers

```python
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
```

### Experiments and optimizers

```python
from sklearn import datasets
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

import retack as rk

ds = datasets.load_boston()

em = rk.Experiment(
    models=[DecisionTreeRegressor], metric_funcs=[mean_absolute_error]
)

results = em.run(ds.data, ds.target)

print(results)

em.set_optimizer(
    "DecisionTreeRegressor",
    rk.OptunaOptimizer,
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
```
