# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 13:34:24 2017

@author: tiwarir
"""
###############################################################################
# change to the appropriate working directory
###############################################################################
from __future__ import division
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

###############################################################################
# chagne the working directory
###############################################################################
os.getcwd()
new_dir = os.path.join("Documents", "ponpare")
os.chdir(new_dir)
os.chdir("..")
###############################################################################
# setting display options
###############################################################################
pd.options.display.max_columns = 100
pd.set_option('expand_frame_repr', False)

import collaborative_filtering_functions as cf 
reload(cf)


# get the dictionaries that maps coupon ids to coupon cluster and coupon cluster
# to its features

def check_accuracy_for_different_k(k=5):
    coupon_id_to_clust_id_dict, coupon_clust_def_dict = cf.get_cluster_info()
    train, test = cf.create_train_test_set(n_users = 1000, seed_value = 10)
    purchase_dict = cf.item_purchased(train, test)
    rating_matrix = cf.create_rating_matrix2(train)
    R_final = cf.create_final_rating_matrix(rating_matrix, n_comp = k)
    recommendations_dict = cf.create_recommendation_for_test_users(test, rating_matrix, R_final)
    return cf.calculate_percentage_accuracy(recommendations_dict, purchase_dict)
    

accuracy = []
for k in range(1,200):
    accuracy.append(check_accuracy_for_different_k(k))
    print k, accuracy

x = range(1,200)
plt.scatter(x, accuracy, marker = 'x', s = 1, c = 'r')
plt.xlabel("No of latent vectors")
plt.ylabel("% accuracy")
x = range(1,200,25)
plt.xticks(x)

n_user = 1000
seed = 10
fname = "train_" + "user_"+ str(n_user) + "_seed_" + str(seed) + ".pkl"
fname = "test_" + "user_"+ str(n_user) + "_seed_" + str(seed) + ".pkl"


# checking the accuracy of the popular item
    
def popular_item_accuracy(train, test):
    train, test = cf.create_train_test_set(n_users = 1000, seed_value = 10)
    purchase_dict = cf.item_purchased(train, test)
    recommendations_pop_item_dict = cf.create_recommendation_based_on_popular_item(train, test)    
    return cf.calculate_percentage_accuracy(recommendations_pop_item_dict, purchase_dict)
    

# created to check if the newer module of creating training and testing is working
# fine
train, test = cf.create_train_test_set(n_users = 1000, seed_value = 10)
train.shape
test.shape
train_user = train.USER_ID_hash.unique()
test_user = test.USER_ID_hash.unique()
len(train_user)
len(test_user)
len([user for user in test_user if user not in train_user])


# created for Sukumar to do the testing
# matching the elements of the rating matrix before and after factorization
coupon_id_to_clust_id_dict, coupon_clust_def_dict = cf.get_cluster_info()
train, test = cf.create_train_test_set(n_users = 1000, seed_value = 10)
coupon_visit_selected_users = cf.create_train_test_set1(n_users = 100, seed_value = 10)
rating_matrix = cf.create_rating_matrix1(coupon_visit_selected_users)
R_final = cf.create_final_rating_matrix(rating_matrix, n_comp = 5)
np.savetxt('rating_matrix.txt', rating_matrix)
np.savetxt('final_rating_matrix.txt', R_final)

# visualization of matrix
train.columns
rating_matrix.shape
R_final.shape
plt.matshow(rating_matrix)
plt.matshow(R_final)
plt.matshow(R_final - rating_matrix)

# number of users and number of coupons
coupon_id_to_clust_id_dict, coupon_clust_def_dict = cf.get_cluster_info()
train, test = cf.create_train_test_set(n_users = 1000, seed_value = 10)
purchase_dict = cf.item_purchased(train, test)
os.listdir(".")  
user_list = pd.read_csv("data/user_list.csv")
user_list.shape    
user_list.columns
len(user_list.USER_ID_hash.unique()) 
coupon_list_train =  pd.read_csv("data/coupon_list_train.csv") 
coupon_list_train.shape
len(coupon_clust_def_dict.keys())   
     

###############################################################################
# data analysis
###############################################################################
import matplotlib.pyplot as plt
# 1. Genere
coupon_list_train =  pd.read_csv("data/coupon_list_train.csv") 
coupon_list_train.GENRE_NAME.value_counts()
len(coupon_list_train.GENRE_NAME.unique())
genre_count = coupon_list_train.GENRE_NAME.value_counts(sort=True)
n_genre = len(genre_count)
xval = np.arange(n_genre)
plt.bar(xval, genre_count)

#2. price rate
bins = [-1,25,50,60,70,80,90,100]
sufs = np.arange(len(bins)-1)
labels = ["price_rate" + str(suf) for suf in sufs]
coupon_list_train['price_rate_cat'] = pd.cut(coupon_list_train.PRICE_RATE, bins = bins, labels = labels)
coupon_list_train.price_rate_cat.value_counts()    
    
# 3. catalogue price
bins = [0, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 1000000]
sufs = np.arange(len(bins)-1)
labels = ["catalog_price" + str(suf) for suf in sufs]
coupon_list_train['price_cat'] = pd.cut(coupon_list_train.CATALOG_PRICE, bins = bins, labels = labels)   
coupon_list_train.CATALOG_PRICE.describe()

# 4. top 10 list in trainig data

top_items = train.groupby(by = 'VIEW_COUPON_ID_hash', as_index = False).sum()
top_items = top_items.sort_values(by = 'PURCHASE_FLG', ascending = False)[:10]
top_items
for i in range(10):
    coupon_id = top_items.VIEW_COUPON_ID_hash.iloc[i]
    purchase_count = top_items.PURCHASE_FLG.iloc[i]
    print coupon_clust_def_dict[coupon_id][0], coupon_clust_def_dict[coupon_id][1], coupon_clust_def_dict[coupon_id][2], purchase_count

user_list = pd.read_csv("data/user_list.csv")
bins = [0,20,30,40,50,60,100]
sufs = np.arange(len(bins)-1)
labels = ["age" + str(suf) for suf in sufs]
user_list['age_cat'] = pd.cut(user_list.AGE, bins = bins, labels = labels)

user_list.age_cat.unique()
len(user_list.age_cat.unique())

train.head()
np.sum(train.PURCHASE_FLG == 0)
np.sum(train.PURCHASE_FLG == 1)

train.shape
test.shape
a = set(train.VIEW_COUPON_ID_hash) 
b = set(test.VIEW_COUPON_ID_hash)
len(a.union(b))

rating_matrix = cf.create_rating_matrix1(train)
np.sum(np.sum(rating_matrix==0.0))

train.columns
train_user = set(train.USER_ID_hash.unique())
test_user = set(test.USER_ID_hash.unique())
test_user - train_user
test_user.difference(train_user)
len(test_user)

###############################################################################
# checking if the cluster assignment is working properly. It works by randomly 
# picking up a coupon id then checking if that coupon id and its respective 
# cluster has the same value. It also check the dictionary where cluster id is mapped 
# to its features.
###############################################################################  
def check_cluster_assignment(coupon_id_to_clust_id_dict, coupon_clust_def_dict):
    n = len(coupon_id_to_clust_id_dict)
    i = np.random.choice(n)
    coupon_list_train = cf.load_coupon_data()
    cf.create_coupon_categorical_variable(coupon_list_train)
    for key, value in coupon_id_to_clust_id_dict.items()[i:i+1]:
        ind_key = coupon_list_train.COUPON_ID_hash == key
        ind_value = coupon_list_train.COUPON_ID_hash == value
        columns_to_keep = ['COUPON_ID_hash', 'GENRE_NAME','price_cat', 'price_rate_cat']        
        print coupon_list_train[columns_to_keep].loc[ind_key]
        print coupon_list_train[columns_to_keep].loc[ind_value]
        print coupon_clust_def_dict[value][0], coupon_clust_def_dict[value][1], coupon_clust_def_dict[value][2]
        print "\n\n"
        
check_cluster_assignment(coupon_id_to_clust_id_dict, coupon_clust_def_dict)
       
################################################################################
# module to check if the purchase items working fine
###############################################################################         
def check_user_purchase():
    n = len(purchase_dict['train'])
    i = np.random.choice(n)
    user_id = purchase_dict['train'].keys()[i]
    ind_user = (train.USER_ID_hash == user_id) & (train.PURCHASE_FLG == 1)
    user_pur = train.loc[ind_user]
    n = len(user_pur)
    print "Training Data"
    for i in range(n):
        print purchase_dict['train'][user_id][i], user_pur.iloc[i,1], user_pur.iloc[i,2]
        
    n = len(purchase_dict['test'])
    i = np.random.choice(n)
    user_id = purchase_dict['test'].keys()[i]
    ind_user = (test.USER_ID_hash == user_id) & (test.PURCHASE_FLG == 1)
    user_pur = test.loc[ind_user]
    n = len(user_pur)
    print "Test Data"
    for i in range(n):
        print purchase_dict['test'][user_id][i], user_pur.iloc[i,1], user_pur.iloc[i,2]
        
check_user_purchase()    
        
        
    
    
 

        
        
        
        



