import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([[4,1],[2,4],[2,3],[3,6],[4,4]])
x2 = np.array([[9,10],[6,8],[9,5],[8,7],[10,8]])

m1 = np.sum(x1, axis=0,dtype=np.float)/x1.shape
m2 = np.sum(x2, axis=0,dtype=np.float)/x2.shape

#print m1
#print x1
#print (x1 - m1)

print np.dot(x1 - m1, x1-m1)