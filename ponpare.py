# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:57:36 2017

@author: tiwarir
"""
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm, beta
from __future__ import division

###############################################################################
# setting the directory
###############################################################################
os.getcwd()
new_dir = os.path.join("Documents", "ponpare")
os.chdir(new_dir)
os.getcwd()
files = os.listdir(".")

###############################################################################
# creating directory if it does not exists (RUN IT ONLY ONCE)
###############################################################################
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

###############################################################################
# unzipping the file (RUN IT ONLY ONCE)
###############################################################################
files = os.listdir(".")

for f in files:
    if f.endswith(".zip"):
        zip_ref = zipfile.ZipFile(f, 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()
        
        
###############################################################################        
# reading the data in the files
###############################################################################

###############################################################################
# User Demographics
###############################################################################
files = os.listdir("data")
fname = 'user_list.csv'
ind = files.index(fname)
fname = os.path.join("data", files[ind])
user_list = pd.read_csv(fname)

# understanding user data
# variables in the data
user_list.columns

# 1. Age
bins = np.arange(0,80,5)
plt.hist(user_list.AGE, bins);
        
# 2. Gender
gender_data = user_list.SEX_ID.value_counts()
xval = np.arange(2)
plt.bar(xval, gender_data, tick_label = ['M', 'F']);

# 3. residence data
resid_data = user_list.PREF_NAME.value_counts(sort = True)
xval = np.arange(10)
plt.bar(xval, resid_data[:10]);

for name in resid_data.index[:10]:
    print(name)

# 4. registration date
reg_date_time  = pd.to_datetime(user_list["REG_DATE"])
reg_date = reg_date_time.dt.date
reg_date_count = reg_date.value_counts()
xval = np.arange(10)
plt.bar(xval, reg_date_count[:10])
for date in reg_date_count.index[:10]:
    print(date)


# 5. user id hash
len(user_list.USER_ID_hash)
len(user_list.USER_ID_hash.unique())

# withdraw date
with_date_time = pd.to_datetime(user_list["WITHDRAW_DATE"])
with_date = with_date_time.dt.date
with_date_count = with_date.value_counts()
xval = np.arange(10)
plt.bar(xval, with_date_count[:10])
for date in with_date_count.index[:10]:
    print(date)

###############################################################################
# User Behavior
###############################################################################
files = os.listdir("data")
fname = 'coupon_visit_train.csv' 
ind = files.index(fname)

fname = os.path.join("data", files[ind])
user_beh = pd.read_csv(fname)

user_beh.columns

# 1. purchase count
purchase_count = user_beh.PURCHASE_FLG.value_counts()
xval = np.arange(2)
plt.bar(xval, purchase_count, tick_label = ['not purchased', 'purchased'])
purchase_count

# 2. percentage of user bahavior
len(user_beh.USER_ID_hash.unique())
user_beh.size
user_beh.shape

###############################################################################
# Coupon list (description)
###############################################################################
files = os.listdir("data")
fname = 'coupon_list_train.csv' 
ind = files.index(fname)

fname = os.path.join("data", files[ind])
coupon_list_train = pd.read_csv(fname)
coupon_list_train.columns
coupon_list_train.shape

# 1.Coupon Genre
genre_count = coupon_list_train.GENRE_NAME.value_counts(sort=True)
n_genre = len(genre_count)
xval = np.arange(n_genre)
plt.bar(xval, genre_count)
genre_count

# 2. price rate
coupon_list_train.PRICE_RATE
bins = np.arange(0,100,10)
plt.hist(coupon_list_train.PRICE_RATE)
plt.hist(coupon_list_train.PRICE_RATE, normed = True);

#  3. catalogue price
coupon_list_train.CATALOG_PRICE
plt.hist(coupon_list_train.CATALOG_PRICE)
plt.hist(coupon_list_train.CATALOG_PRICE, normed = True, log = True)
plt.hist(coupon_list_train.CATALOG_PRICE, log = True)
plt.hist(coupon_list_train.CATALOG_PRICE);

# 4. discount price
coupon_list_train.DISCOUNT_PRICE
plt.hist(coupon_list_train.DISCOUNT_PRICE, log = True)
plt.hist(coupon_list_train.DISCOUNT_PRICE);

# 5. discount period
coupon_list_train.VALIDPERIOD 


###############################################################################
# coupon purchase data
###############################################################################
files = os.listdir("data")
fname = 'coupon_detail_train.csv' 
ind = files.index(fname)

fname = os.path.join("data", files[ind])
coupon_purchase = pd.read_csv(fname)
coupon_purchase.columns
coupon_purchase.shape


###############################################################################
# finding relationship between gender and genre
###############################################################################
# 1. merging purchase table with the 
coupon_purchase.head()
coupon_list_train.head()
user_list.head()

purchase_clist = coupon_purchase.merge(coupon_list_train, how = "left", on = 'COUPON_ID_hash')
purchase_clist_ulist = purchase_clist.merge(user_list, how = "left", on = "USER_ID_hash")
purchase_clist_ulist.shape

# 2. gender and genre relationship
############# female ##########################################################
ind = purchase_clist_ulist.SEX_ID == 'f'
f_genre = purchase_clist_ulist.GENRE_NAME.loc[ind]
f_genre.value_counts()

############## male ###########################################################
ind = purchase_clist_ulist.SEX_ID == 'm'
m_genre = purchase_clist_ulist.GENRE_NAME.loc[ind]
m_genre.value_counts()

###############################################################################
purchase_clist_ulist.shape
purchase_clist_ulist.head()

purchase_clist_ulist.SEX_ID.value_counts()
purchase_clist_ulist.columns
user_list.columns

###############################################################################
###############################################################################
# function to create translation
###############################################################################
###############################################################################



###############################################################################
###############################################################################
# functions to calculate probabilities: part 1
###############################################################################
###############################################################################

prob = {}

user_list.columns

i_columns = [1,4]
user_list.columns[i_columns]

## probability generation
for i in i_columns:
    gen_prob = user_list.iloc[:,i].value_counts()/len(user_list)
    prob[user_list.columns[i]] = gen_prob.to_dict()

## printing probabilities
for key in prob.keys():
    for subkey, value in prob[key].items():
        print key, subkey, value
        

    
###############################################################################
###############################################################################
# functions to calculate continuous probabilities 
###############################################################################
###############################################################################
user_list.columns
plt.hist(user_list.AGE, bins = 20)
mu, std = norm.fit(user_list.AGE)

plt.hist(user_list.AGE, bins=25, normed=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()


coupon_list_train.columns
coupon_list_train.CATALOG_PRICE

alpha, beta, loc, scale = beta.fit(coupon_list_train.CATALOG_PRICE)

###############################################################################
###############################################################################
# functions to calculate joint probabilities
###############################################################################
###############################################################################
req_columns = ["GENRE_NAME", "SEX_ID"]
purchase_clist_ulist[req_columns]

joint_prob = {}
genres = purchase_clist_ulist.GENRE_NAME.unique()
genders = purchase_clist_ulist.SEX_ID.unique()
for genre in genres:
    ind = (purchase_clist_ulist.GENRE_NAME == genre)
    probs = purchase_clist_ulist.SEX_ID[ind].value_counts()/sum(ind)
    joint_prob[genre] = probs.to_dict()
    
for key in joint_prob.keys():
    for subkey, value in joint_prob[key].items():
        print key, subkey, value
        
###############################################################################
###############################################################################
# multiple joint probabilities
###############################################################################
###############################################################################
nested_joint_prob = {}
coupon_cols = ["GENRE_NAME", "SMALL_AREA_NAME"]
user_cols = ["SEX_ID", "small_area_name"]
for ccol in coupon_cols:
    c_variables = purchase_clist_ulist[ccol].unique()
    nested_joint_prob[ccol] = {}
    for ucol in user_cols:
        nested_joint_prob[ccol][ucol] = {}
        for variable in c_variables:
            ind = (purchase_clist_ulist[ccol] == variable)
            probs = purchase_clist_ulist[ucol][ind].value_counts()/sum(ind)
            nested_joint_prob[ccol][ucol][variable] = probs.to_dict()
            
        
for key in nested_joint_prob["GENRE_NAME"]["SEX_ID"].keys():
    for subkey, value in nested_joint_prob["GENRE_NAME"]["small_area_name"][key].items():
        print key, subkey, value


nested_joint_prob["GENRE_NAME"]["SEX_ID"]["健康・医療"]                

###############################################################################
# Creating discrete variable
###############################################################################
user_list.head()

variable = "age"
numbers = range(8)
variables = [variable + str(number) for number in numbers]

pd.cut(user_list.AGE, bins = [0,10,20,30,40,50,60,70,100], labels = variables)
user_list["age_group"] = pd.cut(user_list.AGE, bins = 
         [0,10,20,30,40,50,60,70,100], labels = variables)



      











































