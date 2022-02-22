from SMO import svm
import numpy as np

x = np.array([
  [1,2],
  [2,2],
  [2,1.5],
  [2.5,1.5],
  [1,0],

  [2,3],
  [3,2],
  [3,3],
  [2.5,2.5],
  [4,4]
])
y=np.array([-1,-1,-1,-1,-1,  1,1,1,1,1])

tester = svm(C=1,tol=0.02,kernel="dot")
tester.fit(x,y)
tester.printSelf()
tester.predict(np.array([4,4]))
tester.predict(np.array([1,1]))

for i in x:
  print(i)
  tester.predict(i)