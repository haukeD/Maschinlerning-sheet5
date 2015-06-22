import matplotlib.pyplot as plt
import numpy as np

#start
x1 = np.array([[4,1],[2,4],[2,3],[3,6],[4,4]])
x2 = np.array([[9,10],[6,8],[9,5],[8,7],[10,8]])

x = np.array([x1, x2])

#calc meanValues
m_k = []
m = np.array([0.0,0.0])
for i in range(x.shape[0]):
    mi = np.sum(x[i], axis=0,dtype=np.float)/x.shape[1]
    m_k.append(mi)
    m = m + mi

m = m / len(m_k)
print 'm'
print m
print 'm'

S_W = np.zeros(len(m_k))
S_B = np.zeros(len(m_k))

for k in range(x.shape[0]):
    for n in range(x.shape[1]):
        xD = x[k][n] - m_k[k]
        xT = xD[np.newaxis].transpose()
        S_W = S_W + (xT * xD)
        ###print xD
        ##print (xT * xD)
        ###print m_k[k]
        ###print x[k][n] - m_k[k]
    mD = m_k[k] - m
    mDT = mD[np.newaxis].transpose()
    S_B = S_B + x.shape[1] * (mDT * mD)

S_W = S_W / (x.shape[0] * x.shape[1])
S_B = S_B / (x.shape[0] * x.shape[1])

##print "#####################################"
##print S_W
##print S_B

print np.dot(np.linalg.inv(S_W),S_B)
print np.linalg.eigvals(np.dot(np.linalg.inv(S_W),S_B))

###print (x[0][1])[np.newaxis].transpose()

###A = (x[0][1])[np.newaxis].transpose()#np.array([[1], [2]])
###W = x[0][1]#np.array([10,20])
###print A * W

#help(m_k)
#print count(m_k)

##m1 = np.sum(x1, axis=0,dtype=np.float)/x1.shape
##m2 = np.sum(x2, axis=0,dtype=np.float)/x2.shape
##m = (m1 + m2) / 2

#



##nm.append(m1)
##nm.append(m1)

#nm = np.append(nm,m1,0)
#nm = np.append(nm,m2,0)

##print nm
##print np.sum(x1, axis=0,dtype=np.float)/x1.shape


##print np.hstack((nm, np.atleast_2d(m1).t))

##print np.hstack(m1,m2)

###print m

#print m1
#print x1
#print (x1 - m1)

###sum = np.zeros(shape=(x.shape[2],x.shape[2]))
###for k in range(x.shape[0]):
###    for n in range(x.shape[1]):
###        print np.dot(np.transpose((x[k])[n]),(x[k])[n])
###        z = np.dot(np.transpose((x[k])[n]-m[k]), (x[k])[n]-m[k])
###        print z
###        sum = sum + z
###sum = sum / (x.shape[1])
###print sum

#print np.dot(x1 - m1, x1-m1)


###a = np.array( ((2,3)) )
###b = np.array( ((2),(3)))
###print a*b