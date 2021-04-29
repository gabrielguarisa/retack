import pytest

from retack.experiment import ExperimentManager


def test_create_experiment_manager(model_classes):
    em = ExperimentManager(models=model_classes)
    assert isinstance(em, ExperimentManager)
    assert all([a == b for a, b in zip(em.models, model_classes)])


def test_load_experiment_manager(model_classes):
    em = ExperimentManager.load("tests/files/valid-config.yml")
    assert isinstance(em, ExperimentManager)
    assert all([a == b for a, b in zip(em.models, model_classes)])


def test_invalid_load_experiment_manager():
    with pytest.raises(ImportError):
        _ = ExperimentManager.load("tests/files/invalid-config.yml")
