import Orange
import Orange.evaluation
import Orange.classification
import otto
from importlib import reload
reload(otto)

data = Orange.data.Table("iris")

lr = Orange.classification.LogisticRegressionLearner(C=0.5)
lr.name = "logreg.05"

rlr = otto.RandomizedLearner(lr, p=0.3)
rlr.name = "rand.logreg.05.03"

svm = Orange.classification.SVMLearner(C=1.0)
svm.name = "svm.1"

model = rlr(data)
x = model(data)

res = Orange.evaluation.CrossValidation(data, [lr, svm], k=5)
x = Orange.evaluation.AUC(res)