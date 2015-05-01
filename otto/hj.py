import Orange
import Orange.classification
import otto

lr = Orange.classification.LogisticRegressionLearner(C=0.5)
lr.name = "logreg.05"

rf = Orange.classification.SimpleRandomForestLearner(n_estimators=100)
rf.name = "srf-100"

o = otto.Otto("5k")
# o.dump_cv(rf)
o.dump_cv(lr)
o.report_evaluation()


#
# z = log_loss(ps, data.Y)
# print(z)