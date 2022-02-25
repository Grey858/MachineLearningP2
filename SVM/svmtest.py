from SMO import svm
import numpy as np
import matplotlib.pyplot as plt

x = np.array([
  [1,2.5],
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

tester = svm(C=1,tol=0.5,kernel="sigmoid", max_pases=50)
tester.fit(x,y)
xv,bv = tester.printSelf()
tester.predict(np.array([4,4]))
tester.predict(np.array([1,1]))

for i in x:
  print(i, end=" ")
  tester.predict(i)

plt.scatter(x[:5,0], x[:5,1])
plt.scatter(x[5:,0], x[5:,1])
xv[0]/=-1*xv[1]
bv/=-1*xv[1]
print(xv)
print(bv)
#xvals = np.linspace(0,5,2)
#yvals = xv[0]*xvals+bv
#plt.plot(xvals,yvals)

xrange = np.linspace(0,5,20)
yrange = np.linspace(0,5,20)

pos = list()
neg = list()

for xs in xrange:
  for ys in yrange:
    res = tester.predict(np.array([xs,ys]))
    print(f"Result of prediction: {res}")
    if res > 0:
      pos.append(np.array([xs,ys]))
    else:
      neg.append(np.array([xs,ys]))
pos = np.array(pos)
neg = np.array(neg)

plt.scatter(pos[:,0], pos[:,1], s=4 ,c="green", )
plt.scatter(neg[:,0], neg[:,1], s=4, c="red")

plt.show()