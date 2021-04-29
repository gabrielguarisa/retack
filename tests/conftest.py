import pytest
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC


@pytest.fixture()
def model_classes():
    return [DummyClassifier, SVC]
