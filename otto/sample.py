import Orange
import Orange.evaluation
import os.path
import pickle

pkl_name = "data/otto-train.pkl"
if os.path.exists(pkl_name):
    data = pickle.load(open(pkl_name, "rb"))
else:
    data = Orange.data.Table("data/otto-train.tab")
    pickle.dump(data, open(pkl_name, "wb"))

n = 50000

name = "{:d}k.pkl".format(int(n/1000))
print("Original training set: {}".format(len(data)))
print("Attributes: {}".format(len(data.domain.attributes)))
sampled = Orange.evaluation.sample(data, n)
print("Training size {}, test size {}".format(len(sampled[0]), len(sampled[1])))
pickle.dump(sampled, open("data/{}".format(name), "wb"))
print("Sampled train and test stored in: {}".format(name))
