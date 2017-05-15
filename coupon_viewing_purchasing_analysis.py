# -*- coding: utf-8 -*-
"""
Created on Wed May 03 09:16:15 2017

@author: ravitiwari
"""
###############################################################################
# importing the required libraries
###############################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import pickle
from __future__ import division 

##############################################################################
# setting pandas options
###############################################################################
pd.options.display.max_rows = 100
pd.set_option('expand_frame_repr', False)


###############################################################################
# setting up the working directory
###############################################################################
os.getcwd()
os.chdir("C:\\Users\\ravitiwari\\Documents\\ponpare")
os.listdir(".")


###############################################################################
# using only coupon_visit_train data
###############################################################################

fname = 'data\coupon_visit_train.csv' 
coupon_visit_train = pd.read_csv(fname, sep = ",")
coupon_visit_train.head()

###############################################################################
# preparing list of bought coupons and viewed only coupons
###############################################################################
# bought coupons
ind = coupon_visit_train.PURCHASE_FLG == 1
bought_coupons = coupon_visit_train[ind]["VIEW_COUPON_ID_hash"]
bought_coupons = bought_coupons.unique()

# testing if the bought coupon is working properly
random_coupon = np.random.choice(bought_coupons)
ind = (coupon_visit_train.PURCHASE_FLG == 1) & (coupon_visit_train.VIEW_COUPON_ID_hash == random_coupon)
coupon_visit_train[ind][["PURCHASE_FLG", "VIEW_COUPON_ID_hash"]]

# viewed only coupons
all_coupons = coupon_visit_train["VIEW_COUPON_ID_hash"]
ind = all_coupons.isin(bought_coupons)
viewed_only_coupons = all_coupons[~ind].unique()

# testing if the viewed only coupon is working properly
random_coupon = np.random.choice(viewed_only_coupons)
ind =  coupon_visit_train.VIEW_COUPON_ID_hash == random_coupon
coupon_visit_train[ind][["PURCHASE_FLG", "VIEW_COUPON_ID_hash"]]


###############################################################################
# script to find out number of coupons viewed and bought for each user as 
# well as script to test the results
###############################################################################
view_purchase_dict = defaultdict(lambda: [0,0])

n_rows, n_cols = coupon_visit_train.shape
n_rows

for i in xrange(n_rows):
    user_id = coupon_visit_train.USER_ID_hash.iloc[i]
    coupon_id = coupon_visit_train.VIEW_COUPON_ID_hash.iloc[i]
    if coupon_visit_train.PURCHASE_FLG.iloc[i] == 0:
        view_purchase_dict[(user_id, coupon_id)][0] += 1
    else:
        view_purchase_dict[(user_id, coupon_id)][1] += 1
        

# testing if it is working properly
# test 1: choosing both coupons and users randomly
all_coupons = coupon_visit_train["VIEW_COUPON_ID_hash"].unique()
all_users = coupon_visit_train["USER_ID_hash"].unique()
random_coupon = np.random.choice(all_coupons)
random_user = np.random.choice(all_users)

view_purchase_dict[(random_user, random_coupon)]
ind = (coupon_visit_train.USER_ID_hash == random_user) & (coupon_visit_train.VIEW_COUPON_ID_hash == random_coupon)
coupon_visit_train[ind]

# test 2: choosing coupons randomly and user from the list who has either viewed 
# or bought the coupon
all_coupons = coupon_visit_train["VIEW_COUPON_ID_hash"].unique()
random_coupon = np.random.choice(all_coupons)
ind = coupon_visit_train.VIEW_COUPON_ID_hash == random_coupon
user_viewed_coupons = coupon_visit_train.USER_ID_hash[ind].unique()
random_coupon_user = np.random.choice(user_viewed_coupons)

ind = (coupon_visit_train.USER_ID_hash == random_coupon_user) & (coupon_visit_train.VIEW_COUPON_ID_hash == random_coupon)
coupon_visit_train[ind][["PURCHASE_FLG", "USER_ID_hash", "VIEW_COUPON_ID_hash"]]
random_coupon_user, random_coupon
view_purchase_dict[(random_coupon_user, random_coupon)]

# test 3: choosing a user randomly and coupon from the list of his visited and
# bought coupons
all_users = coupon_visit_train["USER_ID_hash"].unique()
random_user = np.random.choice(all_users)
ind = coupon_visit_train.USER_ID_hash == random_user
user_viewed_coupons = coupon_visit_train.VIEW_COUPON_ID_hash[ind].unique()
random_coupon_user = np.random.choice(user_viewed_coupons)

ind = (coupon_visit_train.USER_ID_hash == random_user) & (coupon_visit_train.VIEW_COUPON_ID_hash == random_coupon_user)
coupon_visit_train[ind][["PURCHASE_FLG", "USER_ID_hash", "VIEW_COUPON_ID_hash"]]
random_user, random_coupon_user
view_purchase_dict[(random_user, random_coupon_user)]

# test 4 randomly pick one key from the dictionay and see if it is giving the 
# right result

keys = view_purchase_dict.keys()
ind = np.random.choice(len(keys))
key = keys[ind]

print key, ":", view_purchase_dict[key]
ind = (coupon_visit_train.USER_ID_hash == key[0]) & (coupon_visit_train.VIEW_COUPON_ID_hash == key[1])
print coupon_visit_train[ind][["PURCHASE_FLG", "USER_ID_hash", "VIEW_COUPON_ID_hash"]]

        
###############################################################################
# saving the coupon viewed and bought information for later use (future
# implementation)
###############################################################################
#file = open("coupon_view_purchase.pkl", "wb")
#pickle.dump(view_purchase_dict, file)
#file.close()

#pkl_file = open('coupon_view_purchase.pkl', 'rb')
#view_purchase_dict = pickle.load("pkl_file")
#pkl_file.close()


###############################################################################
# viewed frequency and purchase frequency
###############################################################################
       
only_viewed = []
purchased = []
for key in view_purchase_dict:
    if view_purchase_dict[key][1] == 0:
        only_viewed.append(view_purchase_dict[key][0])
    else:
        purchased.append(view_purchase_dict[key][1])
    
plt.hist(only_viewed, bins = [1,2,3,4,5,6,7,8,9,10], normed = False, rwidth = 0.9,
         align = 'left')
plt.xlabel("Viewing count of a coupon")
plt.ylabel("Number of users")
plt.yscale('log', nonposy='clip')


plt.hist(purchased, bins = [1,2,3,4,5,6,7,8,9,10], normed = False, rwidth = 0.9,
         align = 'left')
plt.xlabel("Purchase count of a coupon")
plt.ylabel("Number of users")
plt.yscale('log', nonposy='clip')


###############################################################################
# purchase probability after viewing once, twice, thrice etc
# increase view count for views in the range 1 to 10. increase the purchase 
# count whenever the purchase count has non zero value for a given view count
###############################################################################
purchase_count_on_view_count_dict = defaultdict(lambda: [0,0])
n_viewed_bef_purchase = 10
view_count = 0
purchase_count = 0

for i in xrange(1, n_viewed_bef_purchase):
    for key in view_purchase_dict:
        if view_purchase_dict[key][0] == i:
            purchase_count_on_view_count_dict[i][0] += 1
            if view_purchase_dict[key][1] is not 0:
                purchase_count_on_view_count_dict[i][1] += view_purchase_dict[key][1]
                

purchase_prob = []
keys = purchase_count_on_view_count_dict.keys()
for key in  np.sort(keys):
    purchase_prob.append(purchase_count_on_view_count_dict[key][1]/purchase_count_on_view_count_dict[key][0])

# plotting the result
max = 5
x = range(1,max+1)
plt.bar(x, purchase_prob[:max])
plt.xlabel("Coupon view count") 
plt.ylabel("Purchase probability")        

# checking individually for bought once, twice, etc.
n_viewed_bef_purchase = 3
view_count = 0
purchase_count = 0

for key in view_purchase_dict:
    if view_purchase_dict[key][0] == n_viewed_bef_purchase:
        view_count += 1
        if view_purchase_dict[key][1] is not 0:
            purchase_count += view_purchase_dict[key][1]

print view_count
print purchase_count

purchase_probability = purchase_count/view_count
print purchase_probability

    

            

















































