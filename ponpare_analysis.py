# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 08:38:27 2017

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
# setting the directory
###############################################################################
os.getcwd()
new_dir = os.path.join("Documents", "ponpare")
os.chdir(new_dir)
os.getcwd()
files = os.listdir(".")


###############################################################################
# reading the saved file
###############################################################################
#user_purchase = pd.read_pickle("ponpare_user_purchase.pkl")
user_purchase = pd.read_pickle("user_coupon_buying.pkl")

###############################################################################


###############################################################################
# creating subset of data for quick analysis
###############################################################################

#columns_to_use = ["GENRE_NAME", "PRICE_RATE", "CATALOG_PRICE", "ken_name",  "SEX_ID",
#                  "AGE", "PREF_NAME"]

# Checking if the user PREF_NAME and item ken_name are same for all the purchases
# if it is the case than I can analyse by considering each case separetly

len(user_purchase.large_area_name.unique())
len(user_purchase.ken_name.unique())
len(user_purchase.PREF_NAME.unique())
len(user_purchase.ken_name)
sum(user_purchase.ken_name == user_purchase.PREF_NAME)/len(user_purchase.ken_name)
# only 25% of the users PREF_NAME matches the ken_name given on the coupon. 
# Since it would lead to reduction in data by 75%, i have decided to do analysis
# of all the cases together.


columns_to_use = ["GENRE_NAME", "PRICE_RATE", "CATALOG_PRICE",  "SEX_ID",
                  "AGE"]

user_purchase_subset = user_purchase[columns_to_use]
y = user_purchase.PURCHASE_FLG
###############################################################################
# exploratory analysis
###############################################################################

#1. price rate distribution
bins = [0,25,50,60,70,80,90,100]
plt.hist(user_purchase_subset.PRICE_RATE, bins = bins, 
                            histtype='bar', rwidth=0.95);
plt.xlabel("PRICE_RATE")
plt.ylabel("Count")

#2. CATALOG_PRICE distribution
bins = [0, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]

plt.hist(user_purchase_subset.CATALOG_PRICE, bins = bins, 
                            histtype='bar', rwidth=0.95);
plt.xlabel("CATALOG_PRICE (yen)")
plt.ylabel("Count")


#3 GENRE_NAME distribution
genre_count = user_purchase_subset.GENRE_NAME.value_counts(sort=True)
n_genre = len(genre_count)
xval = np.arange(n_genre)
plt.bar(xval, genre_count)
[item for item in genre_count]

#4 SEX_ID distribution
gender_count = user_purchase_subset.SEX_ID.value_counts(sort=True)
n_gender = len(gender_count)
xval = np.arange(n_gender)
plt.bar(xval, gender_count, tick_label = ["f", "m"])

# 5 age distribution
bins = [10,20,30,40,50,60,70,80]
plt.hist(user_purchase_subset.AGE, bins = bins, 
                            histtype='bar', rwidth=0.95);
plt.xlabel("AGE")
plt.ylabel("Count")

##############################################################################
#converting into categorical variable age
# 1.age
i = 0
X_cat = pd.DataFrame(columns = np.arange(5))
bins = [0,20,30,40,50,60,100]
sufs = np.arange(len(bins)-1)
labels = ["age" + str(suf) for suf in sufs]
X_cat[i] = pd.cut(user_purchase_subset.AGE, bins = bins, labels = labels)

# 2. price rate
i = i + 1
bins = [0,25,50,60,70,80,90,100]
sufs = np.arange(len(bins)-1)
labels = ["price_rate" + str(suf) for suf in sufs]
X_cat[i] = pd.cut(user_purchase_subset.PRICE_RATE, bins = bins, labels = labels)

# 3. catalog price
i = i+1
bins = [0, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 1000000]
sufs = np.arange(len(bins)-1)
labels = ["catalog_price" + str(suf) for suf in sufs]
X_cat[i] = pd.cut(user_purchase_subset.CATALOG_PRICE, bins = bins, labels = labels)

# 4. sex_id
i = i+1
X_cat[i] = user_purchase_subset.SEX_ID

# 5. genre_name
i = i+1
X_cat[i] = user_purchase_subset.GENRE_NAME

# 6. purchase probability
#i = i+1
#X_cat[i] = 1
#X_cat.drop([5], axis =1)
y = user_purchase.PURCHASE_FLG 

###############################################################################
# one hot encoding
###############################################################################
y = user_purchase.PURCHASE_FLG 
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X = pd.get_dummies(user_purchase_subset)
X_imp = imp.fit_transform(X)
X_cat_encoded = pd.get_dummies(X_cat)
X_train, X_test, y_train, y_test = train_test_split(X_cat_encoded, y) 

clf = MultinomialNB(alpha = 1)
clf.fit(X_train, y_train)
y_predict = clf.predict_proba(X_test)
X_test

X_cat.head()

##############################################################################
# random forest
###############################################################################
user_purchase_subset
X = user_purchase_subset.dropna(axis = 1, how = 'any')
#imp = Imputer(missing_values='NaN', strategy='median', axis=0)

null_ind = user_purchase.isnull().any(axis=1)

X_encoded = pd.get_dummies(X)


clf = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X_imp, y) 
clf.fit(X_train, y_train)
y_predict = clf.predict_proba(X_test)
y_predict = clf.predict(X_test)
y_predict[:5]
X_test[:5]
y_test[:5]
X_cat.head()
X_train.max()

pd.DataFrame(y_predict , y_test)
ind = y_predict > 0
sum(y_test[ind])

















