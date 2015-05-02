import Orange
import Orange.classification
import sys
import otto
import os.path

lr = Orange.classification.LogisticRegressionLearner(C=0.5)
lr.name = "logreg.05"

rf = Orange.classification.SimpleRandomForestLearner(n_estimators=100)
rf.name = "srf-100"

rlr = otto.RandomizedLearner(lr)
rlr.name = "rand.logreg.05"

rlr = otto.RandomizedLearner(lr, k=50, p=0.5)
rlr.name = "rand.logreg.05.05.50"

svm = Orange.classification.SVMLearner(C=1.0)
svm.name = "svm.1"

knn = Orange.classification.KNNLearner(n_neighbors=11, weights="distance")
knn.name = "knn.11.dist"

learners = {"lr": lr, "rf": rf, "rlr": rlr, "svm": svm, "knn": knn}

d_name, l_name = sys.argv[1:3]
if not os.path.exists("data/{}.pkl".format(d_name)):
    print("Error: data/{}.pkl does not exist".format(d_name))
    sys.exit(0)
if l_name not in learners:
    print("Error: learner {} does not exist.".format(l_name))
    sys.exit(0)

print("Learning on {} with {}.".format(d_name, learners[l_name].name))

o = otto.Otto(d_name)
o.dump_cv(learners[l_name])
o.report_evaluation()
