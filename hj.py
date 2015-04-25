import numpy as np
import Orange
import joblib
import time


class RFLearner(Orange.classification.Learner):
    """
    A random forest classifier, optimized for speed. Trees in the forest
    are constructed with :obj:`SimpleTreeLearner` classification trees.


    Parameters
    ----------

    n_estimators : int, optional (default = 10)
        Number of trees in the forest.

    min_instances : int, optional (default = 2)
        Minimal number of data instances in leaves. When growing the three,
        new nodes are not introduced if they would result in leaves
        with fewer instances than min_instances. Instance count is weighed.

    max_depth : int, optional (default = 1024)
        Maximal depth of tree.

    max_majority : float, optional (default = 1.0)
        Maximal proportion of majority class. When this is
        exceeded, induction stops (only used for classification).

    skip_prob : string, optional (default = "sqrt")
        Data attribute will be skipped with probability ``skip_prob``.

        - if float, then skip attribute with this probability.
        - if "sqrt", then `skip_prob = 1 - sqrt(n_features) / n_features`
        - if "log2", then `skip_prob = 1 - log2(n_features) / n_features`

    seed : int, optional (default = 42)
        Random seed.
    """

    name = 'simple rf'

    def __init__(self, n_estimators=10, min_instances=2, max_depth=1024,
                 max_majority=1.0, skip_prob='sqrt', seed=42):

        self.n_estimators = n_estimators
        self.skip_prob = skip_prob
        self.max_depth = max_depth
        self.min_instances = min_instances
        self.max_majority = max_majority
        self.seed = seed

    def fit_storage(self, data):
        return RFModel(self, data)


class RFModel(Orange.classification.base.Model):
    def s_tree(self, tree, seed, data):
        tree.seed = seed
        return tree(data)

    def __init__(self, learner, data):
        self.estimators_ = []

        if isinstance(data.domain.class_var, Orange.data.DiscreteVariable):
            self.type = 'classification'
            self.cls_vals = len(data.domain.class_var.values)
        elif isinstance(data.domain.class_var, Orange.data.ContinuousVariable):
            self.type = 'regression'
            self.cls_vals = 0
        else:
            assert(False)
        tree = Orange.classification.simple_tree.SimpleTreeLearner(
            learner.min_instances, learner.max_depth,
            learner.max_majority, learner.skip_prob, True)

        parallelizer = joblib.Parallel(n_jobs=-1, max_nbytes=1e1,
                                       verbose=0, backend="multiprocessing")

        print("X", learner.n_estimators)
        tasks = (joblib.delayed(self.s_tree)(tree, learner.seed + i, data)
                 for i in range(learner.n_estimators))
        self.estimators_ = parallelizer(tasks)

    def predict_storage(self, data):
        if self.type == 'classification':
            p = np.zeros((data.X.shape[0], self.cls_vals))
            for tree in self.estimators_:
                p += tree(data, tree.Probs)
            p /= len(self.estimators_)
            return p.argmax(axis=1), p
        elif self.type == 'regression':
            p = np.zeros(data.X.shape[0])
            for tree in self.estimators_:
                p += tree(data)
            p /= len(self.estimators_)
            return p
        else:
            assert(False)

data = Orange.data.Table("adult")

n = 100
rf = RFLearner(n_estimators=n)
srf = Orange.classification.RandomForestLearner(n_estimators=n)

t = time.time()
res = Orange.evaluation.CrossValidation(data, [rf, srf])
print("%3.1fs" % (time.time()-t))
print(Orange.evaluation.AUC(res))
print()

# t = time.time()
# res = Orange.evaluation.CrossValidation(data, [rf])
# print("%3.1fs" % (time.time()-t))
# print(Orange.evaluation.AUC(res))
