# Description: 

import numpy as np
from pylab import *
import orange, orngTest, orngStat, orngEnsemble
data = orange.ExampleTable("testdata_a")

knn = orange.kNNLearner(k=11)
knn.name = "k-NN (k=11)"

classifier = knn(data) # learner learns on data and returns classifier

# test classifier on train data ("manually")
correct = 0.0 # initialization
for sample in data:
    if classifier(sample) == sample.getclass():  # compares classifier result (l.h.s.) with true label (r.h.s.)
        correct += 1
print "Classification accuracy: ", correct/len(data)

# test classifier on train data (using built-in-functions)
learners = [knn]
results  = orngTest.learnAndTestOnLearnData(learners, data)
print "Classification accuracy: ", orngStat.CA(results)[0]

# plot data with true labels
figure()
title('testdata_a with true labels')
xlabel('x')
ylabel('y')
for k in range(data.__len__()):
    sample = data[k]
    if sample.get_class() == '1':
        plot(sample[0].value, sample[1].value, '.r') # class 1: red points
    elif sample.get_class() == '2':
        plot(sample[0].value, sample[1].value, '.b') # class 2: blue points
    else:
        print "unknown class!"


# bagging / boosting
maxIter = 12 # maximum number of bagging / boosting iterations
iter = np.array([i for i in range(1, maxIter+1)])
baggingTrainError  = np.zeros(iter.size)
boostingTrainError = np.zeros(iter.size)
for t in range(1,maxIter+1):
    print "iteration %d" % t
    bagged_knn = orngEnsemble.BaggedLearner(knn, t=t)
    bagged_knn.name = "bagged k-NN"
    boosted_knn = orngEnsemble.BoostedLearner(knn, t=t)
    boosted_knn.name = "boosted k-NN"

    learners = [bagged_knn, boosted_knn] # order not to be changed!
    results  = orngTest.learnAndTestOnLearnData(learners, data)
    for i in range(len(learners)):
        print ("%15s:  %5.3f") % (learners[i].name, orngStat.CA(results)[i])
        # iteration 1 (t) stored at index 0 (t-1)  
        baggingTrainError[t-1] = orngStat.CA(results)[0]  # assumes bagged learner is in first component
        boostingTrainError[t-1] = orngStat.CA(results)[1] # assumes boosted learner is in second component
    
    if t == 1 or t % 4 == 0:
        # plot current classification result
        classifier = bagged_knn(data)
        figure()
        title('testdata_a: bagging: iteration %d' % t) 
        xlabel('x')
        ylabel('y')    
        for k in range(data.__len__()):
            sample = data[k]
            if classifier(sample).value == '1':
                plot(sample[0].value, sample[1].value, '.r') # class 1: red points
            elif classifier(sample).value == '2':
                plot(sample[0].value, sample[1].value, '.b') # class 2: blue points
            else:
                print "unknown class!"

            # plot errors
            if classifier(sample) != sample.get_class():
                plot(sample[0].value, sample[1].value, 'ok') 

        classifier = boosted_knn(data)
        figure()
        title('testdata_a: boosting: iteration %d' % t) 
        xlabel('x')
        ylabel('y')    
        for k in range(data.__len__()):
            sample = data[k]
            if classifier(sample).value == '1':
                plot(sample[0], sample[1].value, '.r') # class 1: red points
            elif classifier(sample).value == '2':
                plot(sample[0], sample[1].value, '.b') # class 2: blue points
            else:
                print "unknown class!"

            # plot errors
            if classifier(sample) != sample.getclass():
                plot(sample[0].value, sample[1].value, 'ok') # 'o': circle, 'k': black
    
# plot
figure()
xlabel('iteration number')
ylabel('classification accuracy')
title('bagging / boosting on testdata_a')
xlim(1, iter.size)
plot (iter, baggingTrainError,'r', iter, boostingTrainError, 'b')
legend(('bagging', 'boosting'), 'lower right')
show()