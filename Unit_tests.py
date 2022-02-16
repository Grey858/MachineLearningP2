import numpy as np
import pandas as pd
import DecisionTree

x=np.array([ 
  [1,0,0],
  [2,1,0],
  [3,0,0],
  [0,2,1],
  [0,0,2],
  [0,1,3]
])
y = np.array([0,0,0,1,1,1])

mytree = DecisionTree.dtree(method="missclassification")
info,sp = mytree.__missclassification__(np.array([0.5,1.5,2.5]),x[:,0],y)
print(f"info: {info}, sp: {sp}")