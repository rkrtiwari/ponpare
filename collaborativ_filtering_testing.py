# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 13:34:24 2017

@author: tiwarir
"""
###############################################################################
# change to the appropriate working directory
###############################################################################
import os
import pandas as pd
import numpy as np
import pickle
os.getcwd()
new_dir = os.path.join("Documents", "ponpare")
os.chdir(new_dir)

###############################################################################
# setting display options
###############################################################################
pd.options.display.max_columns = 100
pd.set_option('expand_frame_repr', False)

import collaborative_filtering_functions as cf 
reload(cf)

# get the dictionaries that maps coupon ids to coupon cluster and coupon cluster
# to its features
        
coupon_id_to_clust_id_dict, coupon_clust_def_dict = cf.get_cluster_info()
train, test = cf.create_train_test_set(n_users = 100, seed_value = 10)
purchase_dict = cf.item_purchased(train, test)
rating_matrix = cf.create_rating_matrix(train) 
rating_matrix = cf.create_rating_matrix1(train)  # rating matrix1 finds the proper weighting factor     
R_final = cf.create_final_rating_matrix(rating_matrix, n_comp = 5)
recommendations_dict = cf.create_recommendation_for_test_users(test, rating_matrix, R_final)
recommendations_pop_item_dict = cf.create_recommendation_based_on_popular_item(train, test)
cf.calculate_percentage_accuracy(recommendations_dict, purchase_dict)
cf.calculate_percentage_accuracy(recommendations_pop_item_dict, purchase_dict)






    

    
     
    

     

    
    
    
    
    
    
    
    























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
        
        
    
    
 

        
        
        
        



