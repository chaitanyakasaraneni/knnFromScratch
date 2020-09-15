# kNN From Scratch

### Introduction

This repository consists of code and example implementations for my medium article on building k-Nearest Neighbors from scratch and evaluating it using k-Fold Cross validation which is also built from scratch.

![Neighbors](https://github.com/chaitanyakasaraneni/knnFromScratch/blob/master/images/neighbors.jpg)
<p align="center">Neighbors (Image Source: <a href="https://www.freepik.com/free-vector/apartment-building-with-people-open-window-spaces_7416533.htm#page=1&query=neighbors&position=2">Freepik</a>)</p>

#### *k*-Nearest Neighbors
*k*-Nearest Neighbors, kNN for short, is a very simple but powerful technique used for making predictions. The principle behind kNN is to use **“most similar historical examples to the new data.”**

#### *k*-Nearest Neighbors in 4 easy steps
 - Choose a value for *k*
 - Find the distance of the new point to each record of training data
 - Get the k-Nearest Neighbors
 - Making Predictions
   - For classification problem, the new data point belongs to the class that most of the neighbors belong to. 
   - For regression problem, the prediction can be average or weighted average of the label of k-Nearest Neighbors

Finally, we evaluate the model using *k*-Fold Cross Validation technique

#### *k*-Fold Cross Validation
This technique involves randomly dividing the dataset into k-groups or folds of approximately equal size. The first fold is kept for testing and the model is trained on remaining k-1 folds.
![kFCV](https://github.com/chaitanyakasaraneni/knnFromScratch/blob/master/images/kFCV.png)
<p align="center">5 fold cross validation. Blue block is the fold used for testing.  (Image Source: <a href="https://scikit-learn.org/stable/modules/cross_validation.html">sklearn documentation</a>)</p>

### Datasets Used

The datasets used here are taken from UCI Machine Learning Repository
 - [Hayes-Roth Dataset](https://archive.ics.uci.edu/ml/datasets/Hayes-Roth)
 - [Car Evaluation Dataset](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
 - [Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer)
 
Car Evaluation and Breast cancer datasets contain text attributes. As we cannot run the classifier on text attributes, we need to convert categorical input features. This is done using `LabelEncoder` of `sklearn.preprocessing`. Label encoder can be applied on a dataframe or a list
**Applying LabelEncoder on entire dataframe**
```
from sklearn import preprocessing

df = pd.DataFrame(data)
df = df.apply(preprocessing.LabelEncoder().fit_transform)
```
**Applying LabelEncoder on a list**
```
labels = preprocessing.LabelEncoder().fit_transform(inputList)
```

#### References
- More info on Cross Validation can be seen [here](https://medium.com/datadriveninvestor/k-fold-and-other-cross-validation-techniques-6c03a2563f1e)
- [kNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [kFold Cross Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
