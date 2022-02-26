from SMO import svm
from SMO import multiple_svm
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
  [4,4],
])

x2 = np.array([
  [2,1.9],
  [2.2,2],
  [2,2.3],
  [1.8,2],
  [2.1,1.7],

  [3,2],
  [3.1,1.5],
  [3,2.5],
  [2.9,1],
  [3.1,0.9],

  [1.3,3],
  [1.1,3],
  [1,3.1],
  [0.7,3],
  [1.1,2.9]
])
x=x2
y=np.array([2,2,2,2,2,  3,3,3,3,3, 4,4,4,4,4])

C=1
tol=0.1
kernel = "polynomial"
gamma = 0.1
r=1
degree = 5

tester = multiple_svm(C=C,tol=tol,kernel=kernel, max_pases=100,num_classes=3, gamma=gamma, r=r, degree=degree)
tester.fit(x,y)
#xv,bv = tester.printSelf()
#tester.predict(np.array([4,4]))
#tester.predict(np.array([1,1]))

for i in x:
  print(i, end=" ")
  tester.predict(i)


#xv[0]/=-1*xv[1]
#bv/=-1*xv[1]
#print(xv)
#print(bv)
#xvals = np.linspace(0,5,2)
#yvals = xv[0]*xvals+bv
#plt.plot(xvals,yvals)

xrange = np.linspace(0,5,40)
yrange = np.linspace(0,5,40)

r=list()
g=list()
b=list()

for xs in xrange:
  for ys in yrange:
    res = tester.predict(np.array([xs,ys]))
    print(f"Result of prediction: {res}")
    if res == 2:
      r.append(np.array([xs,ys]))
    elif res == 3:
      g.append(np.array([xs,ys]))
    elif res == 4:
      b.append(np.array([xs,ys]))
r = np.array(r)
g = np.array(g)
b = np.array(b)

if len(r > 0):
  plt.scatter(r[:,0], r[:,1], s=70 ,c="red", marker="s")
if len(g > 0):
  plt.scatter(g[:,0], g[:,1], s=70 ,c="green", marker="s" )
if len(b > 0):
  plt.scatter(b[:,0], b[:,1], s=70 ,c="blue", marker="s" )


plt.scatter(x[:5,0], x[:5,1], s=90)
plt.scatter(x[5:10,0], x[5:10,1], s=90)
plt.scatter(x[10:,0], x[10:,1], s=90)

plt.title(f"Kernel: {kernel}, C: {C}, Tol: {tol}, Gamma: {gamma}, R: {0.85}")

plt.show()