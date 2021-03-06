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

print(results)
