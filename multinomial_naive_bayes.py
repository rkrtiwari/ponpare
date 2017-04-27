# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:19:58 2017

@author: ravitiwari
"""

import os
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


os.getcwd()
dir(sk.dataset)
iris = datasets.load_iris()
iris.data
iris.target

species_count = pd.value_counts(iris.target)

np.to_array(pd.value_counts(iris.target))

iris.data[:,1]
iris.feature_names[1]
i = 1
plt.hist(iris.data[:,i])
plt.xlabel(iris.feature_names[i])

n_obs, n_features = iris.data.shape
for i in xrange(n_features):
    plt.figure()
    plt.hist(iris.data[:,i])
    plt.xlabel(iris.feature_names[i])
 
plt.bar(np.array(species_count.index), species_count.values);
species_count.plot.bar(); 

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

###############################################################################
# Gaussian Naive Bayes
###############################################################################
clf = GaussianNB()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

metrics.accuracy_score(y_predict, y_test)
metrics.confusion_matrix(y_predict, y_test)

clf.intercept_  # does not work on gaussian data
clf.coef_       # does not work on gaussian data


###############################################################################
# Multinomial naive Bayes
###############################################################################
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
metrics.accuracy_score(y_predict, y_test)
metrics.confusion_matrix(y_predict, y_test)

y_predict

for y_p, y_t in zip(y_predict, y_test):
    print y_p, y_t, y_p == y_t
    
    
clf._joint_log_likelihood
clf.intercept_
clf.coef_

X_train[1,:]

clf.intercept_ + np.dot(clf.coef_, X_train[1,:])


###############################################################################
# creating bins before applying multinomial gaussina
###############################################################################

sufs = np.arange(10)
pre = "slength"

labels = [pre + str(suf) for suf in sufs ]

X
pd.cut(X[:,1], bins = 10, labels = labels)


## converting into categorical variables
i = 0
#X_cat = np.zeros((150,4))
X_cat = pd.DataFrame(columns = np.arange(4))
n_split = 10
sufs = np.arange(n_split)
for feature in iris.feature_names:
    pre = '_'.join(iris.feature_names[1].split()[0:2])
    labels = [pre + str(suf) for suf in sufs ]
    X_cat[i] = pd.cut(X[:,i], bins = n_split, labels = labels)
    i = i+1
 
    
##############################################################################
# converting inton one hot encoding
###############################################################################
X_cat.get_dummies()

X_cat_encoded = pd.get_dummies(X_cat)
X_train, X_test, y_train, y_test = train_test_split(X_cat_encoded, y) 

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
metrics.accuracy_score(y_predict, y_test)
metrics.confusion_matrix(y_predict, y_test)
y_predict

clf.coef_
clf.intercept_



###############################################################################
# training classifier on the multinomial variable
###############################################################################
X_cat 
y 

X_train, X_test, y_train, y_test = train_test_split(X_cat, y) 

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
metrics.accuracy_score(y_predict, y_test)
metrics.confusion_matrix(y_predict, y_test)

y_predict

## now using one hot encoding
X_train[:3,:]

    



















  

