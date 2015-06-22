import matplotlib.pyplot as plt
import numpy as np

#start
x1 = np.array([[4,1],[2,4],[2,3],[3,6],[4,4]])
x2 = np.array([[9,10],[6,8],[9,5],[8,7],[10,8]])

x = np.array([x1, x2])

#calc meanValues
for i in range(x.shape[0]):
    m.append(np.sum(x[i], axis=0,dtype=np.float)/x[i].shape)

print m

##m1 = np.sum(x1, axis=0,dtype=np.float)/x1.shape
##m2 = np.sum(x2, axis=0,dtype=np.float)/x2.shape
##m = (m1 + m2) / 2

#

m = []

##nm.append(m1)
##nm.append(m1)

#nm = np.append(nm,m1,0)
#nm = np.append(nm,m2,0)

##print nm
##print np.sum(x1, axis=0,dtype=np.float)/x1.shape


##print np.hstack((nm, np.atleast_2d(m1).t))

##print np.hstack(m1,m2)

print m

#print m1
#print x1
#print (x1 - m1)

sum = np.zeros(shape=(x.shape[2],x.shape[2]))
for k in range(x.shape[0]):
    for n in range(x.shape[1]):
        print np.dot(np.transpose((x[k])[n]),(x[k])[n])
        z = np.dot(np.transpose((x[k])[n]-m[k]), (x[k])[n]-m[k])
        print z
        sum = sum + z
sum = sum / (x.shape[1])
print sum

#print np.dot(x1 - m1, x1-m1)


a = np.array( ((2,3)) )
b = np.array( ((2),(3)))
print a*b