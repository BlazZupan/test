import Orange
import Orange.classification
import otto

lr = Orange.classification.LogisticRegressionLearner(C=1)
lr.name = "logreg.1"

rf = Orange.classification.SimpleRandomForestLearner(n_estimators=1000)
rf.name = "srf-1000"

o = otto.Otto("5k")
o.dump_cv(rf)
o.report_evaluation()

# x = otto.dump_cv(lr)
# res = otto.test_on_test()
# x = otto.dump_cv(None)

#
# z = log_loss(ps, data.Y)
# print(z)