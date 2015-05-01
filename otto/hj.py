import Orange
import Orange.classification

lr = Orange.classification.LogisticRegressionLearner(C=1)
lr.name = "logreg.1"

rf = Orange.classification.SimpleRandomForestLearner(n_estimators=100)
rf.name = "srf-100"

otto = Otto("5k")
otto.dump_cv(rf)
otto.report_evaluation()

# x = otto.dump_cv(lr)
# res = otto.test_on_test()
# x = otto.dump_cv(None)

#
# z = log_loss(ps, data.Y)
# print(z)