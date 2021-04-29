import pytest

from retack.experiment import ExperimentManager


class CustomModel:
    pass


def test_create_experiment_manager(model_classes):
    em = ExperimentManager(models=model_classes)
    assert isinstance(em, ExperimentManager)
    assert all([a == b for a, b in zip(em.models, model_classes)])


def test_create_experiment_manager_without_models():
    with pytest.raises(ValueError):
        _ = ExperimentManager(models=[])


def test_experiment_manager_load(model_classes):
    em = ExperimentManager.load("tests/files/valid-config.yml")
    assert isinstance(em, ExperimentManager)
    assert all([a == b for a, b in zip(em.models, model_classes)])


def test_experiment_manager_load_invalid_filename():
    with pytest.raises(FileNotFoundError):
        _ = ExperimentManager.load("not/found/invalid-filename.yml")


def test_experiment_manager_load_invalid_model():
    with pytest.raises(ImportError):
        _ = ExperimentManager.load("tests/files/invalid-config.yml")


def test_experiment_manager_load_custom_model():
    em = ExperimentManager.load("tests/files/custom-model.yml")
    assert isinstance(em, ExperimentManager)
    assert em.models[0] == CustomModel
