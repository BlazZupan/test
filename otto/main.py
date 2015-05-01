import Orange
import Orange.classification
import otto

lr = Orange.classification.LogisticRegressionLearner(C=0.5)
lr.name = "logreg.05"

rf = Orange.classification.SimpleRandomForestLearner(n_estimators=100)
rf.name = "srf-100"

rlr = otto.RandomizedLearner(lr)
rlr.name = "rand.logreg.05"

rlr = otto.RandomizedLearner(lr, k=50, p=0.5)
rlr.name = "rand.logreg.05.05.50"

o = otto.Otto("5k")
# o.dump_cv(rf)
o.dump_cv(rlr)
o.report_evaluation()


#
# z = log_loss(ps, data.Y)
# print(z)
