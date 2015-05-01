import Orange
import Orange.evaluation
import otto
from importlib import reload
reload(otto)

data = Orange.data.Table("voting")

lr = Orange.classification.LogisticRegressionLearner(C=0.5)
lr.name = "logreg.05"

rlr = otto.RandomizedLearner(lr, p=0.3)
rlr.name = "rand.logreg.05.03"

model = rlr(data)
x = model(data)

res = Orange.evaluation.CrossValidation(data, [lr, rlr], k=5)
x = Orange.evaluation.AUC(res)