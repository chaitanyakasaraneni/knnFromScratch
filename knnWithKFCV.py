#!/usr/bin/env python
# coding: utf-8

# ## Classification using kNN

# **Import the required libraries**

# In[1]:


import numpy as np
import pandas as pd
import operator
from random import randrange
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')


# **The following cell contains a class of methods to calculate distance between two points using various techniques**

# **Formula to calculate Eucledian distance:**
# 
# <math>\begin{align}D(x, y) = \sqrt{ \sum_i (x_i - y_i) ^ 2 }\end{align}</math>

# **Formula to calculate Manhattan Distance:**
# 
# <math>\begin{align}D(x, y) = \sum_i |x_i - y_i|\end{align}</math>

# **Formula to calculate Hamming Distance:**
# 
# <math>\begin{align}D(x, y) = \frac{1}{N} \sum_i \delta_{x_i, y_i}\end{align}</math>

# In[2]:


class distanceMetrics:
    '''
    Description:
        This class contains methods to calculate various distance metrics
    '''
    def __init__(self):
        '''
        Description:
            Initialization/Constructor function
        '''
        pass
        
    def euclideanDistance(self, vector1, vector2):
        '''
        Description:
            Function to calculate Euclidean Distance
                
        Inputs:
            vector1, vector2: input vectors for which the distance is to be calculated
        Output:
            Calculated euclidean distance of two vectors
        '''
        self.vectorA, self.vectorB = vector1, vector2
        if len(self.vectorA) != len(self.vectorB):
            raise ValueError("Undefined for sequences of unequal length.")
        distance = 0.0
        for i in range(len(self.vectorA)-1):
            distance += (self.vectorA[i] - self.vectorB[i])**2
        return (distance)**0.5
    
    def manhattanDistance(self, vector1, vector2):
        """
        Desription:
            Takes 2 vectors a, b and returns the manhattan distance
        Inputs:
            vector1, vector2: two vectors for which the distance is to be calculated
        Output:
            Manhattan Distance of two input vectors
        """
        self.vectorA, self.vectorB = vector1, vector2
        if len(self.vectorA) != len(self.vectorB):
            raise ValueError("Undefined for sequences of unequal length.")
        return np.abs(np.array(self.vectorA) - np.array(self.vectorB)).sum()
    
    def hammingDistance(self, vector1, vector2):
        """
        Desription:
            Takes 2 vectors a, b and returns the hamming distance
            Hamming distance is meant for discrete-valued vectors, though it is a 
            valid metric for real-valued vectors.
        Inputs:
            vector1, vector2: two vectors for which the distance is to be calculated
        Output:
           Hamming Distance of two input vectors 
        """
        self.vectorA, self.vectorB = vector1, vector2
        if len(self.vectorA) != len(self.vectorB):
            raise ValueError("Undefined for sequences of unequal length.")
        return sum(el1 != el2 for el1, el2 in zip(self.vectorA, self.vectorB))


# In[3]:


class kNNClassifier:
    '''
    Description:
        This class contains the functions to calculate distances
    '''
    def __init__(self,k = 3, distanceMetric = 'euclidean'):
        '''
        Description:
            KNearestNeighbors constructor
        Input    
            k: total of neighbors. Defaulted to 3
            distanceMetric: type of distance metric to be used. Defaulted to euclidean distance.
        '''
        pass
    
    def fit(self, xTrain, yTrain):
        '''
        Description:
            Train kNN model with x data
        Input:
            xTrain: training data with coordinates
            yTrain: labels of training data set
        Output:
            None
        '''
        assert len(xTrain) == len(yTrain)
        self.trainData = xTrain
        self.trainLabels = yTrain

    def getNeighbors(self, testRow):
        '''
        Description:
            Train kNN model with x data
        Input:
            testRow: testing data with coordinates
        Output:
            k-nearest neighbors to the test data
        '''
        
        calcDM = distanceMetrics()
        distances = []
        for i, trainRow in enumerate(self.trainData):
            if self.distanceMetric == 'euclidean':
                distances.append([trainRow, calcDM.euclideanDistance(testRow, trainRow), self.trainLabels[i]])
            elif self.distanceMetric == 'manhattan':
                distances.append([trainRow, calcDM.manhattanDistance(testRow, trainRow), self.trainLabels[i]])
            elif self.distanceMetric == 'hamming':
                distances.append([trainRow, calcDM.hammingDistance(testRow, trainRow), self.trainLabels[i]])
            distances.sort(key=operator.itemgetter(1))

        neighbors = []
        for index in range(self.k):
            neighbors.append(distances[index])
        return neighbors
        
    def predict(self, xTest, k, distanceMetric):
        '''
        Description:
            Apply kNN model on test data
        Input:
            xTest: testing data with coordinates
            k: number of neighbors
            distanceMetric: technique to calculate distance metric
        Output:
            predicted label 
        '''
        self.testData = xTest
        self.k = k
        self.distanceMetric = distanceMetric
        predictions = []
        for i, testCase in enumerate(self.testData):
            neighbors = self.getNeighbors(testCase)
            output= [row[-1] for row in neighbors]
            prediction = max(set(output), key=output.count)
            predictions.append(prediction)
        
        return predictions


# In[4]:


def printMetrics(actual, predictions):
    '''
    Description:
        This method calculates the accuracy of predictions
    '''
    assert len(actual) == len(predictions)
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            correct += 1
    return (correct / float(len(actual)) * 100.0)


# In[5]:


class kFoldCV:
    '''
    This class is to perform k-Fold Cross validation on a given dataset
    '''
    def __init__(self):
        pass
    
    def crossValSplit(self, dataset, numFolds):
        '''
        Description:
            Function to split the data into number of folds specified
        Input:
            dataset: data that is to be split
            numFolds: integer - number of folds into which the data is to be split
        Output:
            split data
        '''
        dataSplit = list()
        dataCopy = list(dataset)
        foldSize = int(len(dataset) / numFolds)
        for _ in range(numFolds):
            fold = list()
            while len(fold) < foldSize:
                index = randrange(len(dataCopy))
                fold.append(dataCopy.pop(index))
            dataSplit.append(fold)
        return dataSplit
    
    
    def kFCVEvaluate(self, dataset, numFolds, *args):
        '''
        Description:
            Driver function for k-Fold cross validation 
        '''
        knn = kNNClassifier()
        folds = self.crossValSplit(dataset, numFolds)
        print("\nDistance Metric: ",*args[-1])
        print('\n')
        scores = list()
        for fold in folds:
            trainSet = list(folds)
            trainSet.remove(fold)
            trainSet = sum(trainSet, [])
            testSet = list()
            for row in fold:
                rowCopy = list(row)
                testSet.append(rowCopy)
                
            trainLabels = [row[-1] for row in trainSet]
            trainSet = [train[:-1] for train in trainSet]
            knn.fit(trainSet,trainLabels)
            
            actual = [row[-1] for row in testSet]
            testSet = [test[:-1] for test in testSet]
            
            predicted = knn.predict(testSet, *args)
            
            accuracy = printMetrics(actual, predicted)
            scores.append(accuracy)

        print('*'*20)
        print('Scores: %s' % scores)
        print('*'*20)        
        print('\nMaximum Accuracy: %3f%%' % max(scores))
        print('\nMean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


# In[6]:


def readData(fileName):
    '''
    Description:
        This method is to read the data from a given file
    '''
    data = []
    labels = []

    with open(fileName, "r") as file:
        lines = file.readlines() 
    for line in lines:
        splitline = line.strip().split(',')
        data.append(splitline)
        labels.append(splitline[-1])
    return data, labels


# ### Hayes-Roth Data

# In[7]:


trainFile = 'Datasets/HayesRoth/hayes-roth.data'

trainData, trainLabel = readData(trainFile)

trainFeatures = []
for row in trainData:
    index = row[1:]
    temp = [int(item) for item in index]
    trainFeatures.append(temp)
    
trainLabels = [int(label) for label in trainLabel]


# **Create an object for k-Fold cross validation class**

# In[8]:


kfcv = kFoldCV()


# **Call the Evaluation function of kFCV class**
# 
# *kfcv.kFCVEvaluate(data, foldCount, neighborCount, distanceMetric)*

# In[9]:
print('*'*20)
print('Hayes Roth Data')


kfcv.kFCVEvaluate(trainFeatures, 10, 3, 'euclidean')


# In[10]:


kfcv.kFCVEvaluate(trainFeatures, 10, 3, 'manhattan')


# In[11]:


kfcv.kFCVEvaluate(trainFeatures, 10, 3, 'hamming')


# ### Car Evaluation Data

# In[12]:


carFile = 'Datasets/CarEvaluation/car.data'

carData, carLabel = readData(carFile)
df = pd.DataFrame(carData)
df = df.apply(preprocessing.LabelEncoder().fit_transform)
carFeatures = df.values.tolist()
carLabels = [car[-1] for car in carFeatures] 


# In[13]:
print('*'*20)
print('Car Evaluation Data')
kfcv.kFCVEvaluate(carFeatures, 10, 3, 'euclidean')


# In[14]:


kfcv.kFCVEvaluate(carFeatures, 10, 3, 'manhattan')


# In[15]:


kfcv.kFCVEvaluate(carFeatures, 10, 3, 'hamming')


# ### Breast Cancer Data

# In[16]:

print('*'*20)
print('Breast Cancer Data')

cancerFile = 'Datasets/BreastCancer/breast-cancer.data'

cancerData, cancerLabel = readData(cancerFile)
cdf = pd.DataFrame(cancerData)
cdf = cdf.apply(preprocessing.LabelEncoder().fit_transform)
cancerFeatures = cdf.values.tolist()
cancerLabels = [cancer[-1] for cancer in cancerFeatures] 


# In[17]:


kfcv.kFCVEvaluate(cancerFeatures, 10, 3, 'euclidean')


# In[18]:


kfcv.kFCVEvaluate(cancerFeatures, 10, 3, 'manhattan')


# In[19]:


kfcv.kFCVEvaluate(cancerFeatures, 10, 3, 'hamming')


# In[ ]:





