import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([[4,1],[2,4],[2,3],[3,6],[4,4]])
x2 = np.array([[9,10],[6,8],[9,5],[8,7],[10,8]])

x = np.array([x1, x2])

##m1 = np.sum(x1, axis=0,dtype=np.float)/x1.shape
##m2 = np.sum(x2, axis=0,dtype=np.float)/x2.shape
##m = (m1 + m2) / 2

m = []

##nM.append(m1)
##nM.append(m1)

#nM = np.append(nM,m1,0)
#nM = np.append(nM,m2,0)

##print nM
##print np.sum(x1, axis=0,dtype=np.float)/x1.shape
for i in range(x.shape[0]):
    m.append(np.sum(x[i], axis=0,dtype=np.float)/x[i].shape)

##print np.hstack((nM, np.atleast_2d(m1).T))

##print np.hstack(m1,m2)

##print nM

#print m1
#print x1
#print (x1 - m1)

for k in range(x.shape[0]):
    sumx1 = 0
    for n in range(x.shape[1]):
        ##print x[k][m]-m[k]
        sumx1 = sumx1 + np.dot((x[k])[n]-m[k],(x[k])[n]-m[k])
    print sumx1

#print np.dot(x1 - m1, x1-m1)