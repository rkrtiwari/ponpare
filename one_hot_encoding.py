# -*- coding: utf-8 -*-
"""
Created on Thu May 04 15:52:44 2017

@author: ravitiwari
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm, beta
from __future__ import division
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer

###############################################################################
# data frame used in the calculation
# visit_user_coupon_detail
###############################################################################

columns_to_use = ['PURCHASE_FLG', 'SEX_ID', 'AGE', 'GENRE_NAME', 'PRICE_RATE',
                  'CATALOG_PRICE', 'VALIDFROM', 'VALIDEND']
user_coupon_purchase_detail = visit_user_coupon_detail[columns_to_use]

###############################################################################
# need to look into the code more carefully. worked for some other data now need 
# to adapt for current data
###############################################################################
# conversion into categorical variables
###############################################################################
# 1.age
i = 0
X_cat = pd.DataFrame(columns = np.arange(5))
bins = [0,20,30,40,50,60,100]
sufs = np.arange(len(bins)-1)
labels = ["age" + str(suf) for suf in sufs]
X_cat[i] = pd.cut(user_purchase_subset_nona.AGE, bins = bins, labels = labels)

# 2. price rate
i = i + 1
bins = [0,25,50,60,70,80,90,100]
sufs = np.arange(len(bins)-1)
labels = ["price_rate" + str(suf) for suf in sufs]
X_cat[i] = pd.cut(user_purchase_subset_nona.PRICE_RATE, bins = bins, labels = labels)

# 3. catalog price
i = i+1
bins = [0, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 1000000]
sufs = np.arange(len(bins)-1)
labels = ["catalog_price" + str(suf) for suf in sufs]
X_cat[i] = pd.cut(user_purchase_subset_nona.CATALOG_PRICE, bins = bins, labels = labels)

# 4. sex_id
i = i+1
X_cat[i] = user_purchase_subset_nona.SEX_ID

# 5. genre_name
i = i+1
X_cat[i] = user_purchase_subset_nona.GENRE_NAME


# convert to numeric using one hot encoding
X_encoded = pd.get_dummies(X_cat)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state = 500)

###############################################################################
# training a classifier and testing its performance
###############################################################################
# 1. naive bayes
clf = MultinomialNB(alpha = 1)
clf.fit(X_train, y_train)
y_predict = clf.predict_proba(X_test)
y_predict_class = clf.predict(X_test)
ind = y_test == 1
for i in np.arange(100):
    print y_predict[i,1]

ind = np.argsort(y_predict[:,1])
y_predict[ind,1]

n = 1000
actual_n = sum(y_test.values[ind[-1]:ind[-1]+n])
actual_n*100/n

#2. Random forest

clf = RandomForestClassifier()
 
clf.fit(X_train, y_train)
y_predict = clf.predict_proba(X_test)
y_predict_class = clf.predict(X_test)


ind = np.argsort(y_predict[:,1])
y_predict[ind,1]

n = 100
actual_n = sum(y_test.values[ind[-1]:ind[-1]+n])
actual_n*100/n

sum(y_test == 1)*100/len(y_test)