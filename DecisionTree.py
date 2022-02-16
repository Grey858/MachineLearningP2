from turtle import left
from unicodedata import category
import numpy as np
import pandas as pd
from sklearn import datasets

class dtree:
  class treenode:
    parent=None
    children = list() # list of child nodes
    data = None
    labels = None
    categories = list()
    leaf = False
    category = -1
    split = -1

    left_dat = None
    right_dat = None
    left_lab = None
    right_lab = None
    
    def __init__(self, data=None, labels = None, parent=None):
      self.parent = parent
      self.data = data
      self.labels = labels
      if len(np.unique(self.labels))<=1:
        print("This node's data is homogenious")
        self.leaf = True
      if self.parent.cutoff > self.labels.shape[0]:
        self.leaf = True

      print("Initializing node")
      return self
    def classify(self, x):
      print("Should return a class or call a child's classify")
      if not self.leaf:
        if x[self.category]<self.split:
          return self.children[0].classify(x)
        else:
          return self.children[1].classify(x)
      # If this node Is a leaf node, then it will not split anything and can return 
      # it's most common class
      else:
        dat_count = dict()
        for i in range(self.data.shape[0]):
          dat_count[self.labels[i]] = dat_count.get(self.labels[i], 0) + 1
        #return the key with the max value = classification with max likelyhood
        return max(dat_count, dat_count.get)
      
    def split_points(self,col):
      #return the split points for this column as a list
      points = list()
      sorted = np.sort(col)
      for i in range(1,len(sorted)):
        if sorted[i-1]!=sorted[i]:
          points.append( (sorted[i-1]!=sorted[i])/2 )
      if len(points)==0:
        points.append(0)
      return points
    def set_split(self):
      max_cat = -1
      max_info = 0
      split_point = -1
      s_func = self.parent.split_func
      for i in self.categories:
        sp = self.split_points(self.data[:,i])
        temp_info, s_point = s_func(sp, self.data[:,i], self.labels)
        #If our current info gain is less than the saved one we split here
        if temp_info > max_info:
          max_info = temp_info
          max_cat = i
          split_point = s_point
      self.category = max_cat
      self.split = split_point

      # If number of nodes in each side of the split is less than a certain amount
      # or if max info is less than a certain amount, this node should be a leaf node
      # otherwise, we need to make two child nodes and split them
      left_ind = np.where(self.data[:,self.category] < self.split)[0]    
      right_ind = np.where(self.data[:,self.category] >= self.split)[0]
      left_dat = self.data[left_ind,:]
      right_dat = self.data[right_ind,:]
      left_lab = self.labels[left_ind]
      right_lab = self.labels[right_ind]
    def leaf_split(self):

  root = None
  split_method = "entropy" #entropy, misclassification, gini
  cutoff = 10 
  split_func = None

  # These should return the info and a split point where info is maximized
  # so for misclassification we will return 1-misclass%
  def __entropy__():
    print("Doing entropy calc")
    return 0.5
  def __gini__():
    print("Doing gini calc")
    return 0.5
  def __missclassification__():
    print("Doing missclass calc")
    return 0.5

  def __init__(self, method="entropy"):
    self.split_method = method
    if method == "entropy":
      self.split_func = self.__entropy__
    elif method == "gini":
      self.split_func = self.__gini__
    elif method == "missclassification":
      self.split_func = self.__missclassification__
  
  # returns numpy array where all values are continuous and
  # cetegorical variables are one-hot encoded. categorical
  # variables must be listed in columns
  def make_buckets(self, df, columns):
    print("Transforming dataframe into categorical data")
    return pd.get_dummies(df, columns=columns).to_numpy()

  def predict(self,x):
    print("predicting data")
  def fit(self, x, y):
    print("fitting to data")