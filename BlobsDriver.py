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
  return tx, ty, tvx, tvy, tex, tey

def get_blobs_hard(tvts = [0.7,0.15,0.15]):
  data = blobs(150, [np.array([3, 4]), np.array([5, 6])], [np.array([[0.7, 0.1], [0.1, 0.43]])] * 2).to_numpy()
  l = data.shape[0]
  tx = data[0:int(l*tvts[0]),:2]
  ty = data[0:int(l*tvts[0]),2:].astype(int).flatten()
  tvx = data[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1]),:2]
  tvy = data[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1]),2:].astype(int).flatten()
  tex = data[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2]),:2]
  tey = data[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2]),2:].astype(int).flatten()
  return tx, ty, tvx, tvy, tex, tey

fp=[-1,0.6]
bag=[True,False]
methods=["missclassification", "entropy"]
data_sizes=[50,10,5,1]
min_infos=[0.0,0.01,0.05,0.2]
num_trees=[1,50,100,150]
tx, ty, tvx, tvy, tex, tey = get_blobs_hard()
print(ty)
import ML_Democracy as MLD

df = pd.DataFrame()
for f in fp:
  for b in bag:
    for m in methods:
      for ds in data_sizes:
        for minf in min_infos:
          for numt in num_trees:

            def dtree_init(x_train, y_train, x_test, y_test):
              dt = dtree(method=m, min_data_size=ds, min_info=minf)
              dt.fit(x_train, y_train)
              test_score = dt.score(x=x_test, y=y_test)
              train_score = dt.score(x=x_train, y=y_train)
              return train_score, test_score, dt
            def predict(model, x):
              temp = model.predict(x)
              return temp

            MLD.set_num_classifications(2)
            # arguments are: x_train, x_test, y_train, y_test
            MLD.set_default_data(tx, tvx, ty, tvy)
            MLD.add_algo(MLD.ML_Algo(dtree_init, predict, "dtree"),numt)
            MLD.train_algos(featureProportion=f, bag=b) # add nullable args for funnel or not
            MLD.current_algos()
            print(f"time taken total: {MLD.train_time()}")
            t = MLD.train_time()
            v1 = MLD.validate_voting(tex, tey, method=0) # change this back to validate, when training set function to vote or not
            v2 = MLD.validate_voting(tex, tey, method=1) # change this back to validate, when training set function to vote or not
            v3 = MLD.validate_voting(tex, tey, method=2) # change this back to validate, when training set function to vote or not
            
            tv1 = MLD.validate_voting(tx, ty, method=0) # change this back to validate, when training set function to vote or not
            tv2 = MLD.validate_voting(tx, ty, method=1) # change this back to validate, when training set function to vote or not
            tv3 = MLD.validate_voting(tx, ty, method=2)
            MLD.remove_all_algos()

            new_row = dict()
            new_row["Feature_Prop"] = f
            new_row["Bagging"] = b
            new_row["Gain Metric"] = m
            new_row["Min Data Size"] = ds
            new_row["Min Info"] = minf
            new_row["Num Trees"] = numt
            new_row["VM1 Train"] = tv1
            new_row["VM2 Train"] = tv2
            new_row["VM3 Train"] = tv2
            new_row["VM1 Test"] = v1
            new_row["VM2 Test"] = v2
            new_row["VM3 Test"] = v3
            df=df.append(new_row, ignore_index=True)

df.to_csv("BlobResults.csv")            
            
# vc largest number of instances such that any possible labeling
# could be classified with a member of our hypothesis space

import matplotlib.pyplot as plt
data = blobs(150, [np.array([3, 4]), np.array([5, 6])], [np.array([[0.7, 0.1], [0.1, 0.43]])] * 2).to_numpy()
print(data.shape)
plt.scatter(x=data[:,0], y=data[:,1], s=5, c=data[:,2])
plt.show()