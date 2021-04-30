import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


@pytest.fixture()
def model_classes():
    return [DummyClassifier, SVC]


@pytest.fixture()
def metric_funcs():
    return [accuracy_score]
