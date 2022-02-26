import csv
import itertools
import math
import numpy as np
import random
import time
import sys

start_time = time.time()
print("Machine Learning Project 2 (Optimized SMO) - Grey Bodi")

with open('spirals.csv', newline='') as f:
    reader = csv.reader(f)
    headings = next(reader)
    data = list(reader)
    data = [list( map(float,i) )[1:4] for i in random.sample(data, 100)] #convert to numbers
    for i in data:
        if i[2] == 0: i[2] = -1
        else: i[2] == 1



def kernel(x, z):
    global kernelName
    global sigma
    global c
    if kernelName == "inner product":
        return np.dot(x,z)
    elif kernelName == "gaussian":
        xm = np.matrix(x)
        zm = np.matrix(z)
        return np.exp((-(np.linalg.norm(xm-zm)**2))/(2*sigma**2))
    elif kernelName == "kernel":
        return math.pow(np.dot(x,z)+c, 2)
    else: print("wrong name!!!!")

def f2(x):
    global b
    global data
    sum = 0
    for i in range(0,m):
        sum += a[i] * data[i][2] * kernel(data[i][0:2], x)
    return sum + b

def smo(C, tol, max_passes, data):
    global a
    global alli
    global b
    global m
    maxf = f2(data[0][0:2])
    #print(f"init maxf: {maxf}")
    i_low = 0
    minf = f2(data[0][0:2])
    i_up = 0
    passes = 0
    totaliters = 0
    while (passes < max_passes):
        num_changed_alphas = 0
        for i in range(0,m):
            xi = data[i][0:2] #always i is i_low, which is max f2
            yi = data[i][2]

            # if (0 < a[i] < C or (yi == 1 and a[i] == C) or (yi == -1 and a[i] == 0)):
            #     blow = f2(data[i_low][0:2])
            #     maxcalc = max(f2(xi), blow)
            #     #print(f"maxcalc: {maxcalc}, f2(xi): {f2(xi)}")
            #     if maxcalc > blow: #changed f
            #         #print("PASSEDDDDDD")
            #         #blow = maxcalc
            #         i_low = i
            #     else: #didn't break
            #         xi = data[i_low][0:2]
            #         yi = data[i_low][2]
            #         i = i_low
            #print(i)

            ei = f2(xi) - yi

            if ((yi*ei < -tol and a[i] < C) or (yi*ei > tol and a[i] > 0)):

                samplelist = list(range(0,m))
                samplelist.pop(i)
                j = random.sample(samplelist, 1)[0] #samples from range 0,m excluding i
                xj = data[j][0:2]
                yj = data[j][2]

                if (0 < a[j] < C or (yj == 1 and a[j] == 0) or (yj == -1 and a[j] == C)):
                    bup = f2(data[i_up][0:2])
                    mincalc = min(f2(xj), bup)
                    if mincalc <bup: #changed f
                        #bup = mincalc
                        i_up = j
                    else: #didn't break
                        xj = data[i_up][0:2]
                        yj = data[i_up][2]
                        j = i_up
                #print(j)
                ej = f2(xj) - yj
                aiold = a[i]
                ajold = a[j]
                l = -1
                h = -1
                if (yi != yj):
                    l = max(0, a[j]-a[i])
                    h = min(C, C+a[j]-a[i])
                else: #equal
                    l = max(0, a[i]+a[j]-C)
                    h = min(C, a[i]+a[j])
                if (l==h):
                    continue
                n = 2 * kernel(xi, xj) - kernel(xi, xi) - kernel(xj, xj)
                if (n >= 0):
                    continue
                a[j] = a[j] - ( (yi * ei * ej) / n)
                #print(f"aj {a[j]}, yi {yi}, ei {ei}, n {n}")
                if a[j] > h: a[j] = h
                elif a[j] < l: a[j] = l
                if (abs(a[j] - ajold) < math.pow(10, -5)):
                    continue
                a[i] = a[i] + yi*yj * (ajold - a[j])
                #compute b's!
                b1 = b - ei - yi * (a[i] - aiold) * kernel(xi, xi) - yj * (a[j] - ajold) * kernel(xi, xj)
                b2 = b - ei - yi * (a[i] - aiold) * kernel(xi, xj) - yj * (a[j] - ajold) * kernel(xj, xj)
                if (a[i] > 0 and a[i] < C):
                    b = b1
                elif (a[j] > 0 and a[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1
                #print(a)


        if (num_changed_alphas == 0):
            passes += 1
            #print(passes)
            totaliters += 1
        else:
            passes = 0
        
    return





m = len(data)
print(m)
print(data[0])


# kernelName = "inner product"
sigma = 1 #for gaussian
c = 2


# a = np.zeros(m)
# alli = 0
# b = 0


# smo(1, .5, 3, data)
# print(a)
# print(b)

# #prediction!
# successes = 0
# for i in data:
#     if f2(i[0:2]) <= 0: prediction = -1
#     else: prediction = 1
#     #print(f"prediction: {prediction}, actual: {i[2]}")
#     if prediction == i[2]:
#         successes += 1
# print(f"Accuracy = {successes/m}")

# print("--- %s minutes ---" % ((time.time() - start_time)/60))
# start_time = time.time()

# a = np.zeros(m)
# alli = 0
# b = 0


# smo(1, .5, 3, data)
# print(a)
# print(b)

# #prediction!
# successes = 0
# for i in data:
#     if f2(i[0:2]) <= 0: prediction = -1
#     else: prediction = 1
#     #print(f"prediction: {prediction}, actual: {i[2]}")
#     if prediction == i[2]:
#         successes += 1
# print(f"Accuracy = {successes/m}")

# print("--- %s minutes ---" % ((time.time() - start_time)/60))
# start_time = time.time()
# a = np.zeros(m)
# alli = 0
# b = 0


# smo(1, .5, 3, data)
# print(a)
# print(b)

# #prediction!
# successes = 0
# for i in data:
#     if f2(i[0:2]) <= 0: prediction = -1
#     else: prediction = 1
#     #print(f"prediction: {prediction}, actual: {i[2]}")
#     if prediction == i[2]:
#         successes += 1
# print(f"Accuracy = {successes/m}")

# print("--- %s minutes ---" % ((time.time() - start_time)/60))
# start_time = time.time()
















# print("GAUSSIAN")
# kernelName = "gaussian"
# a = np.zeros(m)
# alli = 0
# b = 0

# smo(1, .5, 3, data)
# print(a)
# print(b)

# #prediction!
# successes = 0
# for i in data:
#     if f2(i[0:2]) <= 0: prediction = -1
#     else: prediction = 1
#     #print(f"prediction: {prediction}, actual: {i[2]}")
#     if prediction == i[2]:
#         successes += 1
# print(f"Accuracy = {successes/m}")

# print("--- %s minutes ---" % ((time.time() - start_time)/60))
# start_time = time.time()
# kernelName = "gaussian"
# a = np.zeros(m)
# alli = 0
# b = 0

# smo(1, .5, 3, data)
# print(a)
# print(b)

# #prediction!
# successes = 0
# for i in data:
#     if f2(i[0:2]) <= 0: prediction = -1
#     else: prediction = 1
#     #print(f"prediction: {prediction}, actual: {i[2]}")
#     if prediction == i[2]:
#         successes += 1
# print(f"Accuracy = {successes/m}")

# print("--- %s minutes ---" % ((time.time() - start_time)/60))
# start_time = time.time()
# kernelName = "gaussian"
# a = np.zeros(m)
# alli = 0
# b = 0

# smo(1, .5, 3, data)
# print(a)
# print(b)

# #prediction!
# successes = 0
# for i in data:
#     if f2(i[0:2]) <= 0: prediction = -1
#     else: prediction = 1
#     #print(f"prediction: {prediction}, actual: {i[2]}")
#     if prediction == i[2]:
#         successes += 1
# print(f"Accuracy = {successes/m}")

# print("--- %s minutes ---" % ((time.time() - start_time)/60))
# start_time = time.time()

















print("KERNEL")

kernelName = "inner product"
a = np.zeros(m)
alli = 0
b = 0

smo(1, .5, 3, data)
print(a)
print(b)

#prediction!
successes = 0
for i in data:
    if f2(i[0:2]) <= 0: prediction = -1
    else: prediction = 1
    print(f"prediction: {prediction}, actual: {i[2]}")
    if prediction == i[2]:
        successes += 1
print(f"Accuracy = {successes/m}")

print("--- %s minutes ---" % ((time.time() - start_time)/60))
start_time = time.time()


kernelName = "kernel"
a = np.zeros(m)
alli = 0
b = 0

smo(1, .5, 3, data)
print(a)
print(b)

#prediction!
successes = 0
for i in data:
    if f2(i[0:2]) <= 0: prediction = -1
    else: prediction = 1
    #print(f"prediction: {prediction}, actual: {i[2]}")
    if prediction == i[2]:
        successes += 1
print(f"Accuracy = {successes/m}")

print("--- %s minutes ---" % ((time.time() - start_time)/60))
start_time = time.time()

kernelName = "kernel"
a = np.zeros(m)
alli = 0
b = 0

smo(1, .5, 3, data)
print(a)
print(b)

#prediction!
successes = 0
for i in data:
    if f2(i[0:2]) <= 0: prediction = -1
    else: prediction = 1
    #print(f"prediction: {prediction}, actual: {i[2]}")
    if prediction == i[2]:
        successes += 1
print(f"Accuracy = {successes/m}")

print("--- %s minutes ---" % ((time.time() - start_time)/60))
start_time = time.time()






















'''
breakpoints = [round(i*total) for i in [0,1/5,2/5,3/5,4/5,1]] #for 5 fold cross validation. Change to be break points
intensity = 0.5

sumacc = 0
majoriteration = 1
for bp in range(0,len(breakpoints)-1):
    print("\nTRAINING CROSS-VALIDATION CYCLE %d:" % (majoriteration))
    #breakpoints[bp] == current breakpoint
    lowerbound = breakpoints[bp]
    upperbound = breakpoints[bp+1]

    #training part
    traindata = dict() #training data with key=label digit, value=list of each pixel and how many times it was in the digit
    traincountdig = dict() #each digit (key) with how many times the digit occured in training (value)
    totaltrained = 0
    for row in itertools.chain(itertools.islice(data,0,lowerbound), itertools.islice(data,upperbound,total) ): #itertools.islice(data,upperbound,total)
        label = row[0]
        pixels = row.copy()
        pixels.pop(0)
    
        #"normalize" pixel list into 0's and 1's
        normalized = list()
        for i in pixels:
            if i > (256*intensity):
                normalized.append(1)
            else: normalized.append(0)

        if label not in traindata:
            traindata[label] = normalized
            traincountdig[label] = 1
            totaltrained += 1
        else:
            traindata[label] = [a + b for a, b in zip(traindata[label], normalized)]
            traincountdig[label] += 1
            totaltrained += 1


    #evaluating probabilities (P(h)'s needed for naive bayes)
    trainprob = dict()
    for dig in traindata:
        trainprob[dig] = traincountdig[dig] / totaltrained

    #evaluating a probability table (P(pixel|digit))
    #dict trainingdataprobs has key=digit, value=list of probabilities certain pixels can happen given a certain digit P(h|Di)
    #Laplace smoothing is here. Without it is: traindataprobs[dig] = [(traindata[dig][i] / traincountdig[dig]) for i in range(0,len(traindata[dig]))]
    traindataprobs = dict()
    for dig in traindata:
        traindataprobs[dig] = [((traindata[dig][i] + k) / (traincountdig[dig] + k*2)) for i in range(0,len(traindata[dig]))] 
        #2 because values can be 0 or 1, |X|=2

    #testing part
    
    successes = 0
    for row in itertools.islice(data,lowerbound,upperbound):
        label = row[0]
        pixels = row.copy()
        pixels.pop(0)

        #normalize pixel list into 0's and 1's
        normalized = list()
        for i in pixels:
            if i > (256*intensity):
                normalized.append(1)
            else: normalized.append(0)

        nbprobs = {i: math.log(trainprob[i]) for i in trainprob} #getting started with log(P(h)) then can sum the Di logs..
        for pidx in range(0,len(normalized)):
            if (normalized[pidx] == 1): #for each pixel Di that "exists"... (ensures existing)
                #print("passed normal existing!")
                for dig in traindataprobs:
                    if (traindataprobs[dig][pidx] != 0):#adjust for log 0 error. For implimentation without laplace
                        #print("adding the loggy!")
                        nbprobs[dig] += math.log(traindataprobs[dig][pidx], 10)
        #print(nbprobs)
        #print(label)


        guess = max(nbprobs, key=nbprobs.get)

        if guess == label: #success!
            successes += 1

    accuracy = successes / (upperbound-lowerbound)
    print("Accuracy: %.4f" %(accuracy))
    sumacc += accuracy

    majoriteration += 1

print("\n\nAverage accuracy: %.4f" %(sumacc/(majoriteration-1)))



    

'''
