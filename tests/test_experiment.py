import pandas as pd
import pytest
from sklearn.datasets import make_classification

from retack.experiment import Experiment


class CustomModel:
    pass


def _compare_models_dict(result, expected):
    for key, value in expected.items():
        assert key in result
        assert isinstance(value, type(result[key]))


def test_create_experiment(models, metric_funcs):
    em = Experiment(models=models, metric_funcs=metric_funcs)
    assert isinstance(em, Experiment)
    _compare_models_dict(em.models, models)
    assert all([a == b for a, b in zip(em.metric_funcs, metric_funcs)])


def test_create_experiment_without_models():
    with pytest.raises(ValueError):
        _ = Experiment(models=[], metric_funcs=[])


def test_experiment_load(models, metric_funcs):
    em = Experiment.load("tests/files/valid-config.yml")
    assert isinstance(em, Experiment)
    _compare_models_dict(em.models, models)
    assert all([a == b for a, b in zip(em.metric_funcs, metric_funcs)])


def test_experiment_load_invalid_filename():
    with pytest.raises(FileNotFoundError):
        _ = Experiment.load("not/found/invalid-filename.yml")


def test_experiment_load_invalid_model():
    with pytest.raises(ImportError):
        _ = Experiment.load("tests/files/invalid-config.yml")


def test_experiment_load_custom_model():
    em = Experiment.load("tests/files/custom-model.yml")
    assert isinstance(em, Experiment)
    _compare_models_dict(em.models, {"CustomModel": CustomModel()})


def test_experiment_initial_results():
    em = Experiment.load("tests/files/valid-config.yml")
    assert em.results is None


def test_experiment_run():
    em = Experiment.load("tests/files/valid-config.yml")
    X, y = make_classification(random_state=42)
    results = em.run(X, y)
    assert isinstance(results, pd.DataFrame)
    assert isinstance(em.results, pd.DataFrame)
    assert em.results.columns[0] == "accuracy_score"
    assert len(em.results) == 2
