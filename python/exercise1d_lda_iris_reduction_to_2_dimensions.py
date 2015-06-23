# code modified from the following source:

# Code from Chapter 10 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

# The LDA algorithm

from pylab import *
from numpy import *
from scipy import linalg as la


def lda(data, labels, finalDimension):

    nData = shape(data)[0] # number of data points
    nDim = shape(data)[1]  # input dimension
    
    Sw = zeros((nDim,nDim))
    Sb = zeros((nDim,nDim))
    
    C = cov(transpose(data),bias=1) # bias=1: means that covariance is normalized to 1/N instead of 1/(N-1)
    
    # Loop over classes
    classes = unique(labels) # generates set of class labels 
    for i in range(len(classes)): # loop over classes
        # Extract data points of current class
        indices = squeeze(where(labels==classes[i]))
        d = squeeze(data[indices,:]) # d now contains data points of current class
        # compute NORMALIZED class covariance (normalization factor 1/N_k due to bias=1)
        classcov = cov(transpose(d),bias=1) 
        # add UNNORMALIZED class covariance to within-class scatter matrix Sw
        # -> multiply classocv with float(shape(indices)[0])
        # -> then normalized to number of data points (divide by nData)
        Sw += float(shape(indices)[0])/nData * classcov  
    
    print "Within-class scatter:"
    print Sw
    
    # calculate between-class scatter by subtracting within-class scatter from total covariance
    Sb = C - Sw
    print "Between-class scatter:"
    print Sb

    # Compute eigenvalues and eigenvectors of the matrix product of the inverse of Sw and Sb
    # pinv: pseudo-inverse
    eigenvalues, eigenvectors = linalg.eig(dot(linalg.pinv(Sw),Sb))
    
    # sort eigenvalues into order
    indices = argsort(eigenvalues) # argsort sorts from small to large
    
    indices = indices[::-1] # reverses order, i.e. indices now contain eigenvalues from large to small
    eigenvectors = eigenvectors[:,indices]
    eigenvalues = eigenvalues[indices]
    w = eigenvectors[:,:finalDimension]  
    print "eigenvalues:"
    print eigenvalues
    print "eigenvectors:"
    print eigenvectors
    
    print "eigenvectors after dimension reduction: "
    print w
    newData = dot(data,w)
    return newData,w


# main function

data = np.loadtxt('iris_data.txt')
labels = np.loadtxt('iris_labels.txt')
nDim = shape(data)[1]      # input dimension
finalDimension = nDim - 2  # default: just one dimension removed (corresponding to eigenvector with eigenvalue 0) 

# apply LDA and transform points
newData,w = lda(data,labels,finalDimension)
print "newData"
print newData

# plot
xlabel('first dimension of transformed data')
ylabel('second dimension of transformed data')
title('Data after LDA')
plotStyle = ['or', 'ob', 'og']
classes = unique(labels) # generates set of class labels 
for i in range(len(classes)): # loop over classes
    # Extract data points of current class
    indices = squeeze(where(labels==classes[i]))
    d = squeeze(newData[indices,:]) # d now contains data points of current class
    plot(d[:,0],d[:,1],plotStyle[i])

x_min = newData[:,0].min()-1 
x_max = newData[:,0].max()+1
y_min = newData[:,1].min()-1
y_max = newData[:,1].max()+1    
xlim((x_min, x_max))
ylim((y_min, y_max))
show()
