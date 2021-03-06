import ML_Democracy as MLD
from DecisionTree import dtree
import numpy as np
import pandas as pd

# Sample x and y data I made up, when x3 > x1 and x2 label
# is 0. When x1 and x2 > x3 then label is 1
x = np.array([
  [0,0,3],
  [1,0,4],
  [2,3,1],
  [3,3,0],
  [0,2,5],
  [1,0,4],
  [4,2,0]
])
y=np.array([0,0,1,1,0,0,1])


# need to define a couple functions to send to MLD
# so that it knows how to fit the trees and how
# to use them to predict and vote

# Init method should fit a model to a dataset and 
# return the model's desired score metric on train and test
# and also return a reference to that model
def dtree_init(x_train, y_train, x_test, y_test):
    dt = dtree(method="Missclassification", min_data_size=30, min_info=0.5)
    dt.fit(x_train, y_train)
    test_score = dt.score(x=x_test, y=y_test)
    train_score = dt.score(x=x_train, y=y_train)
    return train_score, test_score, dt
  

# This function wont usually change, but I wrote the
# Library to be used with cross platform libraries and
# sklearn uses model.predict(), other libraries may differ
# but this function should just call a function that gives
# our dtree a set of input datapoints and then records the
# list of output classifications as an np-like array
def predict(model, x):
  temp = model.predict(x)
  return temp

# Binary classification, This function is used for 
# Two of the weighted voting methods
MLD.set_num_classifications(2)

# arguments are: x_train, x_test, y_train, y_test
MLD.set_default_data(x[:4], x[4:], y[:4], y[4:])
MLD.add_algo(MLD.ML_Algo(dtree_init, predict, "dtree"),10)
MLD.train_algos() # add nullable args for funnel or not
MLD.current_algos()
MLD.validate_voting(x[4:], y[4:], method=0) # change this back to validate, when training set function to vote or not
MLD.validate_voting(x[4:], y[4:], method=1) # change this back to validate, when training set function to vote or not
MLD.validate_voting(x[4:], y[4:], method=2) # change this back to validate, when training set function to vote or not

