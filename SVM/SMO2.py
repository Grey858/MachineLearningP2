from random import randint
import pandas as pd
import numpy as np
import math
import time


class svm:
  kernel=None

  def getfx(self,xi):
    tot=0
    for j in range(self.x.shape[0]):
      tot+=self.a[j]*self.y[j]*self.kernel(self.x[j],xi) 
    tot += self.b
    return tot 
  
  def select_j(self,i,big):
    r=-1
    while r<0 or r==i:
      r = randint(0,big-1)
    return r
  
  def get_bounds(self,yi,yj,ai,aj,C):
    L=0
    H=0
    if yi!=yj:
      L=max(0,aj-ai)
      H=min(C,C+aj-ai)
    else:
      L=max(0,ai+aj-C)
      H=min(C,ai+aj)
    return L,H

  def clip(self,a,h,l):
    temp=a
    if temp>h:
      temp=h
    if a<l:
      temp=l
    return temp

  def get_b(self, b, Ei,Ej, yi,yj, newai,oldai, newaj,oldaj, xi,xj, kernel, C):
    #print(f"Calc b: b {b}, Ei {Ei}, Ej {Ej}, yi {yi}, yj {yj}, newai {newai}, oldai {oldai}, newaj {newaj}, oldaj {oldaj}, xi {xi}, xj {xj}")
    #print()
    temp=b
    b1 = b-Ei - yi*(newai-oldai)*kernel(xi,xi) - yj*(newaj-oldaj)*kernel(xi,xj)
    b2 = b-Ej - yi*(newai-oldai)*kernel(xi,xj) - yj*(newaj-oldaj)*kernel(xj,xj)
    if 0<newai<C:
      temp=b1
    elif 0<newaj<C:
      temp=b2
    else:
      temp=(b1+b2)/2.0
    return temp

  def predict(self,x):
    temp = self.getfx(x)
    print(f"Fx: {temp}")
    if(temp>0):
      return 1
    elif(temp<=0):
      return -1

  def printSelf(self):
    print("A coeficients: ")
    print(self.a)
    print(f"b: {self.b}")
    print(f"C: {self.C}")
    print(f"tol: {self.tol}")
    print("Support vectors: ")
    temp=np.zeros(self.x.shape[1])
    for i,ai in enumerate(self.a):
      if ai>0:
        print(f"a{i}: {self.a[i]}, x{i}: {self.x[i]}, y{i}: {self.y[i]}")
        temp+=self.x[i]*self.a[i]*self.y[i]
    print(f"coefs: {temp} + {self.b}")
    return temp,self.b
  
  def dot(self,v1, v2):
    return np.dot(v1,v2)
  def rbf(self,v1, v2):
    temp = np.sum(np.power((v1-v2),2))
    return math.exp(-self.gamma*temp)
  def sigmoid(self,v1,v2):
    temp = np.dot(v1,v2)
    return math.tanh(self.gamma * temp + self.r)
  def polynomial(self,v1,v2):
    return math.pow(np.dot(v1,v2)+self.r,self.degree)

  def __init__(self, C=2, tol=0.95, max_pases=100, kernel="dot", degree=2, r=1, gamma=0.05, time_cutoff=1000):
    print("initializing svm")
    self.C = C
    self.tol = tol
    self.max_passes = max_pases
    self.degree = degree
    self.r = r
    self.gamma = gamma
    self.time_cutoff=time_cutoff

    if kernel == "dot":
      self.kernel = self.dot
    elif kernel == "rbf": #Not yet implemented
      self.kernel = self.rbf
    elif kernel == "polynomial":
      self.kernel = self.polynomial
    elif kernel == "sigmoid":
      self.kernel = self.sigmoid
    else:
      raise Exception(f"{kernel} is Not a valid kernel")
   
  def fit(self, x, y):
    self.x=x
    self.y=y
    self.a = np.zeros(x.shape[0])
    self.b = 0
    passes = 0
    start = time.time()

    num_changed_alphas=0
    examineAll=1

    while num_changed_alphas>0 or examineAll==1:#passes<self.max_passes and time.time()<start+self.time_cutoff:
      num_changed_alphas=0
      

      if(num_changed_alphas == 0):
        passes=passes+1
      else:
        passes=0

  def examineExample(self, i):
    Ei = self.getfx(self.x[i]) - self.y[i]

    if (self.y[i]*Ei < -self.tol and self.a[i] < self.C) or (self.y[i]*Ei>self.tol and self.a[i]>0):
      j = self.select_j(i,self.y.shape[0])
      Ej = self.getfx(self.x[j]) - self.y[j]
      oldai = self.a[i]
      oldaj = self.a[j]

      L,H = self.get_bounds(self.y[i],self.y[j],self.a[i],self.a[j],self.C)

      if L==H:
        return 0
      n = 2*self.kernel(self.x[i],self.x[j])-self.kernel(self.x[i],self.x[i])-self.kernel(self.x[j],self.x[j])
      if n>=0:
        return 0
      newaj = oldaj-( self.y[j] * ( Ei-Ej ) ) / n
      newaj = self.clip(newaj,H,L)
      if abs(newaj-oldaj)<0.00001:
        return 0
      newai = oldai + self.y[i]*self.y[j]*(oldaj-newaj)
      b = self.get_b(self.b,  Ei,Ej,  self.y[i],self.y[j],  newai,oldai, newaj,oldaj, self.x[i],self.x[j], self.kernel, self.C)
      num_changed_alphas += 1
      self.b=b
      self.a[i]=newai
      self.a[j]=newaj


#class multiple_svm:
