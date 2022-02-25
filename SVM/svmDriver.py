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
from sklearn.model_selection import train_test_split

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
  y = df["income"].to_numpy().flatten()
  df = df.drop(["income"], axis=1)
  cats = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
  
  #df = dtree.integer_mapping(None,df,cats)
  #print(df.head())
  df = dtree.make_dummies(None, df, cats)
  
  data = df.to_numpy()[:10000,:]
  #data = StandardScaler().fit_transform(data)
  #data=PCA(n_components=0.9).fit_transform(data)

  y=y[0:10000]
  l = data.shape[0]
  tx = data[0:int(l*tvts[0])]
  ty = y[0:int(l*tvts[0])].astype(int).flatten()
  tvx = data[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1])]
  tvy = y[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1])].astype(int).flatten()
  tex = data[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2])]
  tey = y[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2])].astype(int).flatten()
  return tx, ty, tvx, tvy, tex, tey

def get_MNIST(size=1000):
  from tensorflow import keras
  TRAIN_LEN = 10000
  TEST_LEN  = 2000

  # Data to test with
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  # Scale images to the [0, 1] range
  x_train = x_train.astype("float32") / 255
  x_test = x_test.astype("float32") / 255

  #pc = PCA(n_components=0.9)
  x_train = np.reshape(x_train, (60000, 28*28))
  x_test = np.reshape(x_test, (10000, 28*28))
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

def get_cars(tvts=[0.7,0.15,0.15]):
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

  l = data.shape[0]
  tx = data[0:int(l*tvts[0])]
  ty = y[0:int(l*tvts[0])].astype(int).flatten()
  tvx = data[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1])]
  tvy = y[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1])].astype(int).flatten()
  tex = data[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2])]
  tey = y[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2])].astype(int).flatten()
  return tx, ty, tvx, tvy, tex, tey

def get_flowers(tvts=[0.7,0.15,0.15]):
  df = pd.read_csv("iris_data.csv")
  df = dtree.integer_mapping(None,df,["species"])
  df=df.sample(frac=1)
  y = df["species"].to_numpy().flatten()
  df = df.drop(["species"], axis=1)
  #df = dtree.make_dummies(None, df, cats)
  
  data = df.to_numpy()
  #data = StandardScaler().fit_transform(data)
  #data=PCA(n_components=0.9).fit_transform(data)

  l = data.shape[0]
  tx = data[0:int(l*tvts[0])]
  ty = y[0:int(l*tvts[0])].astype(int).flatten()
  tvx = data[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1])]
  tvy = y[int(l*tvts[0]):int(l*tvts[0])+int(l*tvts[1])].astype(int).flatten()
  tex = data[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2])]
  tey = y[int(l*tvts[1]):int(l*tvts[1])+int(l*tvts[2])].astype(int).flatten()
  return tx, ty, tvx, tvy, tex, tey


filename = "svmBlob"
tx, ty, tex, tey = get_spirals()
deeta = pd.DataFrame()

tester = svm(C=5,tol=0.05,kernel="polynomial", max_pases=100, time_cutoff = 120, gamma=2, degree=7)
tester.fit(tx,ty)
tester.printSelf()


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





def get_accuracy(deeta):
  tester = svm(C=5,tol=0.05,kernel="rbf", max_pases=50, time_cutoff = 60, gamma=3)
  tester.fit(tx,ty)
  new_row = dict()
  #new_row["VM1 Train"] = tv1
  #new_row["VM2 Train"] = tv2
  #new_row["VM3 Train"] = tv2
  new_row["VM1 Test"] = 0
  new_row["VM2 Test"] = 1
  new_row["VM3 Test"] = 2
  new_row["Train Time"] = 3
  deeta=deeta.append(new_row, ignore_index=True)


