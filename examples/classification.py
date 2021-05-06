from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from retack.experiment import Experiment

ds = datasets.load_iris()

em = Experiment(
    models=[SVC(), RandomForestClassifier(), AdaBoostClassifier()],
    metric_funcs=[accuracy_score],
)

results = em.run(ds.data, ds.target)

print(results)
