import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from DecisionTree import dtree
from dataset import blobs
from dataset import spirals
import math
import time
from SMO import svm
from SMO import multiple_svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def get_blobs():
  data = blobs(200, [np.array([1, 2]), np.array([5, 6])], [np.array([[0.25, 0], [0, 0.25]])] * 2).to_numpy()
  for i in range(data.shape[0]):
    if data[i,2]==0:
      data[i,2]=-1

  train, test = train_test_split(data, train_size=0.75,shuffle=True)
  tx = train[:,:2]
  ty = train[:,2]
  tex = test[:,:2]
  tey = test[:,2]

  return tx,ty,tex,tey

def get_blobs_hard():
  data = blobs(150, [np.array([3, 4]), np.array([5, 6])], [np.array([[0.7, 0.1], [0.1, 0.43]])] * 2).to_numpy()
  for i in range(data.shape[0]):
    if data[i,2]==0:
      data[i,2]=-1

  train, test = train_test_split(data, train_size=0.75,shuffle=True)
  tx = train[:,:2]
  ty = train[:,2]
  tex = test[:,:2]
  tey = test[:,2]

  return tx,ty,tex,tey

def get_spirals(shuffle=True):
  data = spirals(n=1000, cycles=2, sd=0.05).to_numpy()
  for i in range(data.shape[0]):
    if data[i,2]==0:
      data[i,2]=-1

  train, test = train_test_split(data, train_size=0.75,shuffle=shuffle)
  tx = train[:,:2]
  ty = train[:,2]
  tex = test[:,:2]
  tey = test[:,2]

  return tx,ty,tex,tey

def get_adult(size=1000):
  df = pd.read_csv("adult_data.csv")
  df=df.sample(frac=1)
  df["income"] = pd.Categorical(df["income"]).codes
  y = df["income"].to_numpy().flatten()[:size]

  

  df = df.drop(["income"], axis=1)
  cats = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
  
  df = dtree.make_dummies(None, df, cats)
  data = df.to_numpy()[:size,:]

  for i in range(y.shape[0]):
    if y[i]==0:
      y[i]=-1
    else:
      y[i]=1

  print(y[0:10])
  tx = data[:int(3*size/4),:]
  tex = data[int(3*size/4):,:]
  ty = y[:int(3*size/4)]
  tey = y[int(3*size/4):]

  return tx, ty, tex, tey

def get_MNIST(size=1000):
  from tensorflow import keras
  TRAIN_LEN = size
  TEST_LEN  = int(size/6)

  # Data to test with
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  # Scale images to the [0, 1] range
  x_train = x_train.astype("float32") / 255
  x_test = x_test.astype("float32") / 255

  pc = PCA(n_components=0.9)
  x_train = np.reshape(x_train, (60000, 28*28))
  x_test = np.reshape(x_test, (10000, 28*28))
  print(x_test.shape)
  x_train = pc.fit_transform(x_train)
  print(x_train.shape)
  print(x_train[0])
  x_test = pc.transform(np.reshape(x_test, (10000, 28*28)))

  print(f'original shape upon import, x: {x_train.shape}, y: {y_train.shape}')

  x_train = x_train[:TRAIN_LEN]
  y_train = y_train[:TRAIN_LEN].astype(int)
  x_test = x_test[:TEST_LEN]
  y_test = y_test[:TEST_LEN].astype(int)

  return x_train, y_train, x_test, y_test

def get_cars():
  cols=["buy","maint","doors","persons","lug","safety","acc"]
  buy_dict = {"vhigh":4, "high":3, "med":2, "low":1}
  maint_dict = {"vhigh":4, "high":3, "med":2, "low":1}
  doors_dict = {"2":2, "3":3, "4":4, "5more":5}
  persons_dict = {"2":2, "4":4, "more":5}
  lug_boot_dict = {"small":1, "med":2, "big":3}
  safety_dict =  {"low":1, "med":2, "high":3}
  acc_dict={"unacc":0,"acc":1,"good":2,"vgood":3}
  mdict={"buy":buy_dict,"maint":maint_dict,"doors":doors_dict,"persons":persons_dict,"lug":lug_boot_dict,"safety":safety_dict,"acc":acc_dict}

  df = pd.read_csv("car_data.csv")
  for i in cols:
    df[i]=df[i].map(mdict[i])
  df=df.sample(frac=1)
  y = df["acc"].to_numpy().flatten()
  df = df.drop(["acc"], axis=1)
  #df = dtree.make_dummies(None, df, cats)
  
  data = df.to_numpy()
  #data = StandardScaler().fit_transform(data)
  #data=PCA(n_components=0.9).fit_transform(data)
  t = int(3*data.shape[0]/4)

  tx = data[:t]
  ty = y[:t].astype(int).flatten()
  tex = data[t:]
  tey = y[t:].astype(int).flatten()
  return tx, ty, tex, tey

def get_flowers():
  df = pd.read_csv("iris_data.csv")
  df = dtree.integer_mapping(None,df,["species"])
  df=df.sample(frac=1)
  y = df["species"].to_numpy().flatten()
  df = df.drop(["species"], axis=1)
  #df = dtree.make_dummies(None, df, cats)
  
  data = df.to_numpy()
  #data = StandardScaler().fit_transform(data)
  #data=PCA(n_components=0.9).fit_transform(data)

  t = int(3*data.shape[0]/4)

  tx = data[:t]
  ty = y[:t].astype(int).flatten()
  tex = data[t:]
  tey = y[t:].astype(int).flatten()
  return tx, ty, tex, tey


filename = "svm_mnist"
tx, ty, tex, tey = get_MNIST()
deeta = pd.DataFrame()

#tester = svm(C=5,tol=0.05,kernel="polynomial", max_pases=100, time_cutoff = 120, gamma=2, degree=7)
#tester.fit(tx,ty)
#tester.printSelf()

def graphStuff():
  xrange = np.linspace(-2,2,40)
  yrange = np.linspace(-2,2,40)
  pos = list()
  neg = list()
  for xs in xrange:
    for ys in yrange:
      res = tester.predict(np.array([xs,ys]))
      #print(f"Result of prediction: {res}")
      if res > 0:
        pos.append(np.array([xs,ys]))
      else:
        neg.append(np.array([xs,ys]))
  pos = np.array(pos)
  neg = np.array(neg)

  if len(pos>0):
    plt.scatter(pos[:,0], pos[:,1], s=80 ,c="green", marker='s')
  if len(neg>0):
    plt.scatter(neg[:,0], neg[:,1], s=80, c="red", marker='s')


  plt.scatter(tx[:,0], tx[:,1],c="yellow")

  plt.scatter(tex[:,0], tex[:,1], c="blue")
  plt.title("Kernel: polynomial, C=5, Tolerance 0.05, Degree: 7")

  plt.show()





def get_accuracy(deeta, x, y, C, kernel, max_passes, time_cutoff, gamma, degree, r):

  chunk = int(x.shape[0]/5)
  accuracies = list()
  print(f"Getting accuracy y: {y[0:5]}")
  for i in range(5):
    xte = x[i*chunk:(i+1)*chunk]
    yte = y[i*chunk:(i+1)*chunk]

    xt = np.concatenate((x[0:i*chunk], x[(i+1)*chunk:]), axis=0)
    yt = np.concatenate((y[0:i*chunk], y[(i+1)*chunk:]), axis=0)

    tester = multiple_svm(C=C,tol=0.05,kernel=kernel, max_pases=max_passes, time_cutoff = time_cutoff, gamma=gamma, degree=degree, r=r, num_classes=10)
    tester.fit(xt,yt)

    tot=0
    for xi in range(yte.shape[0]):
      temp = tester.predict(xte[xi])
      if temp == yte[xi]:
        tot+=1
    acc = tot/yte.shape[0]
    accuracies.append(acc)
    print(f"acc {i}: {acc}")
  accuracies = np.array(accuracies)


  new_row = dict()
  new_row["Accuracy"] = np.mean(accuracies)
  new_row["Std Deviation"] = np.std(accuracies)
  new_row["C"] = C
  new_row["kernel"] = kernel
  new_row["max_passes"] = max_passes
  new_row["time_cutoff"] = time_cutoff
  new_row["gamma"] = gamma
  new_row["degree"] = degree
  new_row["r"] = r
  
  deeta=deeta.append(new_row, ignore_index=True)
  deeta.to_csv(filename+"_temp.csv") 
  return np.mean(accuracies), deeta

C=[1,2,3,4,5]
max_pases=[50,100,200]

kernel=["rbf","polynomial", "dot"] 
gamma=[1,2,3,4]
degree=[2,3,4] 
r=[1,0.85,1.15]

time_cutoff = [30,60,120,60*20] 


bestScore = 0
bestParams = list()
bc = C[0]
for i,c in enumerate(C):
  temp, deeta = get_accuracy(deeta,tx,ty,c,kernel[0],max_pases[0],time_cutoff[0], gamma[0], degree[0], r[0])
  if temp>bestScore:
    bc=c
    bestScore = temp
    bestParams = [c,kernel[0],max_pases[0],time_cutoff[0], gamma[0], degree[0], r[0]]
print(f"Best score: {bestScore} with params {bestParams}")

bmp = max_pases[0]
for i,mp in enumerate(max_pases):
  temp, deeta = get_accuracy(deeta,tx,ty,bc,kernel[0],mp,time_cutoff[0], gamma[0], degree[0], r[0])
  if temp>bestScore:
    bmp=mp
    bestScore = temp
    bestParams = [bc,kernel[0],mp,time_cutoff[0], gamma[0], degree[0], r[0]]
print(f"Best score: {bestScore} with params {bestParams}")

bkern = kernel[0]
bdeg = degree[0]
bgam = gamma[0]

for k in kernel:

  if(k == "rbf"):
    print("Checking rbf")
    for g in gamma:
      temp, deeta = get_accuracy(deeta,tx,ty,bc,k,bmp,time_cutoff[0], g, bdeg, r[0])
      if temp>bestScore:
        bkern=k
        bgam = g
        bestScore = temp
        bestParams = [bc,k,bmp,time_cutoff[0], g, bdeg, r[0]]
    print(f"Best score: {bestScore} with params {bestParams}")
  if(k == "dot"):
    print("Checking dot")
    temp, deeta = get_accuracy(deeta,tx,ty,bc,k,bmp,time_cutoff[0], bgam, bdeg, r[0])
    if temp>bestScore:
      bkern=k
      bestScore = temp
      bestParams = [bc,k,bmp,time_cutoff[0], bgam, bdeg, r[0]]
    print(f"Best score: {bestScore} with params {bestParams}")
  if(k == "polynomial"):
    print("Checking polynomial")
    for d in degree:
      temp, deeta = get_accuracy(deeta,tx,ty,bc,k,bmp,time_cutoff[0], bgam, d, r[0])
      if temp>bestScore:
        bkern=k
        bdeg = d
        bestScore = temp
        bestParams = [bc,k,bmp,time_cutoff[0], bgam, d, r[0]]
    print(f"Best score: {bestScore} with params {bestParams}")
btime = time_cutoff[0]
for t in time_cutoff:
  temp, deeta = get_accuracy(deeta,tx,ty,bc,bkern,bmp,t, bgam, bdeg, r[0])
  if temp>bestScore:
    bmp=mp
    bestScore = temp
    bestParams = [bc,bkern,bmp,t, bgam, bdeg, r[0]]
print(f"Best score: {bestScore} with params {bestParams}")

deeta.to_csv(filename+"_Results.csv") 

