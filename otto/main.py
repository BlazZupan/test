import Orange
import Orange.classification
import sys
import otto
import os
import re

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

knn = Orange.classification.KNNLearner(n_neighbors=100, weights="distance")
knn.name = "knn.100.dist"

rknn = otto.RandomizedLearner(knn, k=10, p=0.3)
rknn.name = "rand.knn.100.03.dist"

gbl = otto.GradientBoostingLearner(n_estimators=500)
gbl.name = "gbl.500"


learners = {"lr": lr, "rf": rf, "rlr": rlr, "svm": svm, "knn": knn,
            "rknn": rknn, "gbl": gbl}

if len(sys.argv) == 1:
    print("{} data".format(sys.argv[0]))
    print("{} data learner".format(sys.argv[0]))
    print("   where:")
    p = re.compile("(?P<name>[\dkM]+)\.pkl")
    names = [p.match(name).groupdict()["name"]
             for name in os.listdir("data") if p.match(name)]
    print("   data in [{}]".format(", ".join(names)))
    print("   learner in [{}]".format(", ".join(learners.keys())))
    sys.exit(0)

if len(sys.argv) == 2:
    d_name = sys.argv[1]
    o = otto.Otto(d_name)
    o.report_evaluation()
else:
    l_name = sys.argv[1:3]
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
