from turtle import left
from unicodedata import category
import numpy as np
import pandas as pd
from sklearn import datasets

class dtree(object):
  class treenode(object):
    parent=None
    children = list() # list of child nodes
    data = None
    labels = None
    categories = list()
    leaf = False
    category = -1
    split = -1
    
    def __init__(self, data=None, labels = None, parent=None, categories = None):
      self.parent = parent
      self.data = data
      self.labels = labels
      self.categories = categories
      # If we only have 1 class
      if len(np.unique(self.labels))<=1:
        print("This node's data is homogenious")
        self.leaf = True
      # If there is not enough data to make a statistically significant split
      if self.parent.cutoff > self.labels.shape[0]:
        self.leaf = True
      # If there are no categories left to split by
      if len(self.categories)==0:
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
      return np.array(points)
    def set_split(self):
      if(self.leaf):
        return
      max_cat = -1
      max_info = 0
      split_point = -1
      split_points = list() # Needed to see if we only have 1 split point then we remove the categorie
      s_func = self.parent.split_func
      for i in self.categories:
        sp = self.split_points(self.data[:,i])
        temp_info, s_point = s_func(sp, self.data[:,i], self.labels)
        # if the information gained by this potential category is close
        # to nothing then we may want to remove it
        if(temp_info < self.parent.info_cutoff):
          self.categories.remove(i)
          continue
        # If our current info gain is less than the saved one we split here
        if temp_info > max_info:
          max_info = temp_info
          max_cat = i
          split_point = s_point
          split_points = sp
        
      self.category = max_cat
      self.split = split_point
      #if we didn't find a way to split the data meaningfully then this node should be a leaf
      if(max_cat == -1):
        self.leaf=True
        return
      # if the category only has 1 split point then there is no reason to split on
      # this category again so we can remove it from the list 
      if len(split_points)<2:
        self.categories.remove(max_cat)
      # If number of nodes in each side of the split is less than a certain amount
      # or if max info is less than a certain amount, this node should be a leaf node
      # otherwise, we need to make two child nodes and split them
      left_ind = np.where(self.data[:,self.category] < self.split)[0]    
      right_ind = np.where(self.data[:,self.category] >= self.split)[0]
      left_dat = self.data[left_ind,:]
      right_dat = self.data[right_ind,:]
      left_lab = self.labels[left_ind]
      right_lab = self.labels[right_ind]
      self.children.append(type(self)(data=left_dat, labels=left_lab, parent=self.parent, categories=self.categories))
      self.children.append(type(self)(data=right_dat, labels=right_lab, parent=self.parent, categories=self.categories))
      self.children[0].set_split()
      self.children[1].set_split()
  root = None
  split_method = "entropy" #entropy, misclassification, gini
  cutoff = 10 
  info_cutoff = 0.01
  split_func = None

  # These should return the info and a split point where info is maximized
  # so for misclassification we will return 1-misclass%
  def __entropy__():
    print("Doing entropy calc")
    return 0.5
  def __gini__():
    print("Doing gini calc")
    return 0.5
  def __missclassification__(self, s_points, data, labels):
    l_types = np.unique(labels)
    info=0
    s_point = -1
    for i in s_points:
      #split the data by s_points
      left_ind = np.where(data <  i)[0]    
      right_ind = np.where(data>= i)[0]
      left_dat = data[left_ind]
      right_dat = data[right_ind]
      left_lab = labels[left_ind]
      right_lab = labels[right_ind]
      
      dat_count = dict()
      for j in range(left_lab.shape[0]):
        dat_count[left_lab[j]] = dat_count.get(left_lab[j], 0) + 1
      l_max = dat_count[max(dat_count)]
      dat_count = dict()
      for j in range(right_lab.shape[0]):
        dat_count[right_lab[j]] = dat_count.get(right_lab[j], 0) + 1
      r_max = dat_count[max(dat_count)]
      temp_info = (r_max + l_max)/data.shape[0]
      
      if(temp_info > info):
        info = temp_info
        s_point = i
      #find the % classified correctly in each side
    if(info==0):
      print("error, info of 0 found, hit enter to continue")
      input()
    return info, s_point

  def __init__(self, method="entropy", min_data_size = 10, min_info = 0.01):
    self.split_method = method
    if method == "entropy":
      self.split_func = self.__entropy__
    elif method == "gini":
      self.split_func = self.__gini__
    elif method == "missclassification":
      self.split_func = self.__missclassification__
    self.cutoff = min_data_size
    self.info_cutoff = min_info

  # returns numpy array where all values are continuous and
  # cetegorical variables are one-hot encoded. categorical
  # variables must be listed in columns
  def make_dummies(self, df, columns):
    print("Transforming dataframe into categorical data")
    return pd.get_dummies(df, columns=columns).to_numpy()
  def integer_mapping(self, df, columns):
    for i in columns:
      df[i] = df[i].rank(method='dense', ascending=False).astype(int)
    return df
  def predict(self,x):
    print("Predicting data")
    predictions = list()
    for i in x:
      predictions.append(self.root.classify(x))
    return np.array(predictions)
  def fit(self, x, y):
    self.root = self.treenode(data=x, labels=y, parent=self, categories=np.arange(0,y.shape[0],1,int))
    self.root.set_split()