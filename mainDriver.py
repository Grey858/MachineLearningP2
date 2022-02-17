from threading import stack_size
import pandas as pd
import numpy as np
#from sklearn.tree import DecisionTreeClassifier
from DecisionTree import dtree
from dataset import blobs
from dataset import spirals
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
import time

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

def get_spirals(tvts=[0.7,0.15,0.15]):
  data = spirals(n=1000, cycles=2, sd=0.05).to_numpy()
  print(data.shape)
  l = data.shape[0]
  tx = data[0:int(l*tvts[0]),:2]
  ty = data[0:int(l*tvts[0]),2:].astype(int).flatten()
  tvx = data[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1]),:2]
  tvy = data[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1]),2:].astype(int).flatten()
  tex = data[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2]),:2]
  tey = data[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2]),2:].astype(int).flatten()
  return tx, ty, tvx, tvy, tex, tey

def get_adult(tvts=[0.7,0.15,0.15]):
  df = pd.read_csv("adult_data.csv")
  df["income"] = pd.Categorical(df["income"]).codes
  y = df["income"].to_numpy().flatten()
  df = df.drop(["income"], axis=1)
  cats = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
  
  df = dtree.integer_mapping(None,df,cats)
  print(df.head())
  input()
  #df = dtree.make_dummies(None, df, cats)
  
  data = df.to_numpy()
  #data = StandardScaler().fit_transform(data)
  #data=PCA(n_components=0.9).fit_transform(data)
  print(data.shape)
  print(data.shape)
  #y=y[0:1000]
  l = data.shape[0]
  tx = data[0:int(l*tvts[0])]
  ty = y[0:int(l*tvts[0])].astype(int).flatten()
  tvx = data[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1])]
  tvy = y[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1])].astype(int).flatten()
  tex = data[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2])]
  tey = y[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2])].astype(int).flatten()
  return tx, ty, tvx, tvy, tex, tey

def get_MNIST():
  from tensorflow import keras
  TRAIN_LEN = 30000
  TEST_LEN  = 2500

  # Data to test with
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  # Scale images to the [0, 1] range
  x_train = x_train.astype("float32") / 255
  x_test = x_test.astype("float32") / 255

  #pc = PCA(n_components=0.9)
  x_train = np.reshape(x_train, (60000, 28*28))
  #print(x_train.shape)
  #x_train = pc.fit_transform(x_train)
  #print(x_train.shape)
  #x_test = pc.transform(np.reshape(x_test, (10000, 28*28)))

  print(f'original shape upon import, x: {x_train.shape}, y: {y_train.shape}')

  x_train = x_train[:TRAIN_LEN]
  x_val = x_test[int(TEST_LEN/2):]
  x_test = x_test[:int(TEST_LEN/2)]
  y_train = y_train[:TRAIN_LEN]
  y_val = y_test[int(TEST_LEN/2):]
  y_test = y_test[:int(TEST_LEN/2)]
  return x_train, y_train, x_val, y_val, x_test, y_test


bag=[True,False]
methods=["missclassification", "entropy"]
data_sizes=[200,100,50]
min_infos=[0.05]
num_trees=[50,100,150]
filename = "adult_cat"
tx, ty, tvx, tvy, tex, tey = get_adult()
fp=list()
num_fet = math.log(tx.shape[1]+1, 2)
n=2
while(n<num_fet):
  fp.append(n/tx.shape[1]+0.001)
  n*=2
fp.append(num_fet/tx.shape[1]+0.001)
print(num_fet)
print(fp)
fp=[0.1,0.3,0.5,-1]
#mytree = dtree(method="missclassification", min_data_size=20, min_info=0.05, depth_cutoff=5)
#mytree.fit(tx, ty)
#print(mytree.score(tx,ty))
#input("Real code time?")

import ML_Democracy as MLD

deeta = pd.DataFrame()


def get_accuracy(f, b, m, ds, minf, numt, deeta):
  def dtree_init(x_train, y_train, x_test, y_test):
    #ftime=0
    #tetime=0
    #trtime=0 
    
    #st=time.time()
    dt = dtree(method=m, min_data_size=ds, min_info=minf)
    dt.fit(x_train, y_train)
    #ftime=time.time()-st
    #st=time.time()
    test_score = dt.score(x=x_test, y=y_test)
    #tetime=time.time()-st
    #st=time.time()
    train_score = dt.score(x=x_train, y=y_train)
    #trtime=time.time()-st

    #print(f"fit time: {ftime}, train time: {tetime}, test time, {trtime}")

    return train_score, test_score, dt
  def predict(model, x):
    #st=time.time()
    temp = model.predict(x)
    #print(f"pred time: {time.time()-st}")
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
  
  #tv1 = MLD.validate_voting(tx, ty, method=0) # change this back to validate, when training set function to vote or not
  #tv2 = MLD.validate_voting(tx, ty, method=1) # change this back to validate, when training set function to vote or not
  #tv3 = MLD.validate_voting(tx, ty, method=2)
  MLD.remove_all_algos()

  new_row = dict()
  new_row["Feature_Prop"] = f
  new_row["Bagging"] = b
  new_row["Gain Metric"] = m
  new_row["Min Data Size"] = ds
  new_row["Min Info"] = minf
  new_row["Num Trees"] = numt
  #new_row["VM1 Train"] = tv1
  #new_row["VM2 Train"] = tv2
  #new_row["VM3 Train"] = tv2
  new_row["VM1 Test"] = v1
  new_row["VM2 Test"] = v2
  new_row["VM3 Test"] = v3
  deeta=deeta.append(new_row, ignore_index=True)

  print(deeta.head())
  return (v1+v2+v3)/3, deeta

bestScore = 0
bestParams = list()
bf = 0
for i,f in enumerate(fp):
  temp, deeta = get_accuracy(f, bag[0], methods[0], data_sizes[0], min_infos[0], num_trees[0], deeta)
  if temp>bestScore:
    bf=f
    bestScore = temp
    bestParams = [f, bag[0], methods[0], data_sizes[0], min_infos[0], num_trees[0]]
deeta.to_csv(filename+"_results.csv")  
print(f"Best score: {bestScore} with params {bestParams}")
print(bestParams)

bb=0
for i,b in enumerate(bag):
  temp, deeta = get_accuracy(bf, b, methods[0], data_sizes[0], min_infos[0], num_trees[0], deeta)
  if temp>bestScore:
    bb=b
    bestScore = temp
    bestParams = [bf, b, methods[0], data_sizes[0], min_infos[0], num_trees[0]]
deeta.to_csv(filename+"_results.csv")  
print(f"Best score: {bestScore} with params {bestParams}")
print(bestParams)

bm=0
for i,m in enumerate(methods):
  temp, deeta = get_accuracy(bf, bb, m, data_sizes[0], min_infos[0], num_trees[0], deeta)
  if temp>bestScore:
    bm=m
    bestScore = temp
    bestParams = [bf, b, m, data_sizes[0], min_infos[0], num_trees[0]]
bds=0


deeta.to_csv(filename+"_results.csv")  
print(f"Best score: {bestScore} with params {bestParams}")
print(bestParams)


for i,ds in enumerate(data_sizes):
  temp, deeta = get_accuracy(bf, bb, m, ds, min_infos[0], num_trees[0], deeta)
  if temp>bestScore:
    bds=ds
    bestScore = temp
    bestParams = [bf, b, m, ds, min_infos[0], num_trees[0]]

deeta.to_csv(filename+"_results.csv")  
print(f"Best score: {bestScore} with params {bestParams}")
print(bestParams)

bminf = 0
for i,minf in enumerate(min_infos):
  temp, deeta = get_accuracy(bf, bb, m, ds, minf, num_trees[0], deeta)
  if temp>bestScore:
    bminf=minf
    bestScore = temp
    bestParams = [bf, b, m, ds, minf, num_trees[0]]


deeta.to_csv(filename+"_results.csv")  
print(f"Best score: {bestScore} with params {bestParams}")
print(bestParams)

bnumt=0
for i,numt in enumerate(num_trees):
  temp, deeta = get_accuracy(bf, bb, m, ds, minf, numt, deeta)
  if temp>bestScore:
    bnumt=numt
    bestScore = temp
    bestParams = [bf, b, m, ds, minf, numt]

print(f"Best score: {bestScore} with params {bestParams}")
print(bestParams)

deeta.to_csv(filename+"_results.csv")            
            