import pandas as pd
import numpy as np
#from sklearn.tree import DecisionTreeClassifier
from DecisionTree import dtree
from dataset import blobs

def get_blobs(tvts = [0.7,0.15,0.15]):
  data = blobs(200, [np.array([1, 2]), np.array([5, 6])], [np.array([[0.25, 0], [0, 0.25]])] * 2).to_numpy()
  l = data.shape[0]
  tx = data[0:int(l*tvts[0]),:2]
  ty = data[0:int(l*tvts[0]),2:].astype(int).flatten()
  tvx = data[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1]),:2]
  tvy = data[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1]),2:].astype(int).flatten()
  tex = data[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2]),:2]
  tey = data[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2]),2:].astype(int).flatten()

  #print(tvy)
  #input("tvy")

  return tx, ty, tvx, tvy, tex, tey


def dtree_init(x_train, y_train, x_test, y_test):
  dt = dtree(method="missclassification")
  dt.fit(x_train, y_train)
  test_score = dt.score(x=x_test, y=y_test)
  train_score = dt.score(x=x_train, y=y_train)
  return train_score, test_score, dt
def predict(model, x):
  temp = model.predict(x)
  return temp


fp=[-1,0.6]
bag=[True,False]
tx, ty, tvx, tvy, tex, tey = get_blobs()
print(ty)
import ML_Democracy as MLD

for f in fp:
  for b in bag:
    MLD.set_num_classifications(2)
    # arguments are: x_train, x_test, y_train, y_test
    MLD.set_default_data(tx, tvx, ty, tvy)
    MLD.add_algo(MLD.ML_Algo(dtree_init, predict, "dtree"),10)
    MLD.train_algos(featureProportion=f, bag=b) # add nullable args for funnel or not
    MLD.current_algos()
    MLD.validate_voting(tex, tey, method=0) # change this back to validate, when training set function to vote or not
    MLD.validate_voting(tex, tey, method=1) # change this back to validate, when training set function to vote or not
    MLD.validate_voting(tex, tey, method=2) # change this back to validate, when training set function to vote or not
    MLD.remove_all_algos()
# vc largest number of instances such that any possible labeling
# could be classified with a member of our hypothesis space

import matplotlib.pyplot as plt
data = blobs(200, [np.array([1, 2]), np.array([5, 6])], [np.array([[0.25, 0], [0, 0.25]])] * 2).to_numpy()
print(data.shape)
plt.scatter(x=data[:,0], y=data[:,1], s=5, c=data[:,2])
plt.show()