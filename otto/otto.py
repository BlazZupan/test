import Orange
import Orange.classification
import os.path
import pickle
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
                                backend="multiprocessing")

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
        for name in files:
            print("%-30s %6.4f %6.4f" % (name,
                                        self.test_on_cv(name),
                                        self.test_on_test(name)))


def min_max(p):
    return np.maximum(np.minimum(p, 1 - 1e-15), 1e-15)


def log_loss(ps, actual):
    return np.array(np.sum(np.log(min_max(ps[np.arange(ps.shape[0]), actual.astype(np.int)])))) / (-len(actual))