import Orange
import Orange.classification
import pickle
import sklearn
import sklearn.cross_validation as skl_cross_validation
import numpy as np
import os

from joblib import Parallel, delayed


class Otto:
    def __init__(self, name):
        self.name = name
        self.data, self.test = pickle.load(open("data/%s.pkl" % name, "rb"))
        if os.path.exists("data/%s-indices.pkl" % name):
            self.ind = pickle.load(open("data/%s-indices.pkl" % name, "rb"))
        else:
            self.ind = self.create_cv_indices()
            pickle.dump(self.ind, open("data/%s-indices.pkl" % name, "wb"))

    def create_cv_indices(self, k=10):
        y = self.data.Y.copy().flatten()
        indices = skl_cross_validation.StratifiedKFold(
            y, k, shuffle=True
        )
        return indices

    def cv(self, learner, ind_train, ind_test):
        train, test = self.data[ind_train], self.data[ind_test]
        print(len(train), len(test))
        model = learner(train)
        return model(test, 1)

    def dump_cv(self, learner, n_jobs=-1):
        parallelizer = Parallel(n_jobs=n_jobs, max_nbytes=1e15,
                                backend="threading", verbose=20)

        tasks = (delayed(self.cv)(learner, a, b) for a, b in self.ind)
        entries = parallelizer(tasks)
        ps_cv = np.vstack(entries)

        model = learner(self.data)
        ps = model(self.test, 1)

        pickle.dump((ps, ps_cv),
                    open("res/%s-%s.pkl" % (self.name, learner.name), "wb"))

    def test_on_test(self, name):
        ps, _ = pickle.load(open("res/%s.pkl" % (name), "rb"))
        return log_loss(ps, self.test.Y)

    def test_on_cv(self, name):
        _, cv = pickle.load(open("res/%s.pkl" % (name), "rb"))
        actual = np.hstack([self.data[ind].Y for _, ind in self.ind])
        return log_loss(cv, actual)

    def report_evaluation(self):
        files = [name[:-4] for name in os.listdir("res")
                 if name[-4:] == ".pkl" and self.name in name]

        res = [(self.test_on_cv(name), self.test_on_test(name), name)
               for name in files]
        res.sort()
        for cv, tst, name in res:
            print("%-30s %6.4f %6.4f" % (name, cv, tst))


def min_max(p):
    return np.maximum(np.minimum(p, 1 - 1e-15), 1e-15)


def log_loss(ps, actual):
    return np.array(np.sum(np.log(min_max(ps[np.arange(ps.shape[0]),
                                             actual.astype(np.int)])))) \
           / (-len(actual))


class RandomizedLearner(Orange.classification.Learner):
    """Ensamble learning through randomization of data domain."""
    def __init__(self, learner, k=3, p=0.1):
        super().__init__()
        self.k = k
        self.learner = learner
        self.name = "rand " + self.learner.name
        # a function to be used for random attribute subset selection
        self.selector = Orange.preprocess.fss.SelectRandomFeatures(k=p)

    def fit_storage(self, data):
        """Returns a bagged model with randomized regressors."""
        models = []
        for epoch in range(self.k):
            sample = self.selector(data)  # data with a subset of attributes
            models.append(self.learner(sample))
        model = BaggedModel(data.domain, models)
        model.name = self.name
        return model


class BaggedModel(Orange.classification.Model):
    """Bootstrap aggregating classifier."""
    def __init__(self, domain, models):
        super().__init__(domain)
        self.models = models  # a list of predictors

    def predict_storage(self, data, ret=Orange.classification.Model.Value):
        """Given data instances returns predicted probabilities."""
        y_hats = np.array([m(data, 1) for m in self.models]).mean(axis=0)
        return y_hats


class GradientBoostingLearner(Orange.classification.SklLearner):
    __wraps__ = sklearn.ensemble.GradientBoostingClassifier
    name = 'gbc'

    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_depth=3, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.supports_multiclass = True

