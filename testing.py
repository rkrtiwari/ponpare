# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:17:30 2017

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
import time

###############################################################################
# chagne the working directory
###############################################################################
os.getcwd()
new_dir = os.path.join("C:\\","USERS", "tiwarir", "Documents", "ponpare", "comparison_content_collaborative_filtering")
os.chdir(new_dir)
os.listdir(".")
###############################################################################
# setting display options
###############################################################################
pd.options.display.max_columns = 100
pd.set_option('expand_frame_repr', False)

import collaborative_filtering_functions as cf 
reload(cf)

import content_filtering_functions as conf
reload(conf)



# checking how many are from the training set
# k = 5
# calculate the accuracy
coupon_id_to_clust_id_dict, coupon_clust_def_dict = cf.get_cluster_info()
train, test = cf.create_train_test_set(n_users = 1000, seed_value = 10)
purchase_dict = cf.item_purchased(train, test)
view_purchase_dict = cf.item_viewed_purchased(train, test)
rating_matrix = cf.create_rating_matrix1(train)
R_final = cf.create_final_rating_matrix(rating_matrix, n_comp = 5)
cf_recommendations_dict = cf.create_recommendation_for_test_users(test, rating_matrix, R_final)
cf.calculate_percentage_accuracy(cf_recommendations_dict, purchase_dict)


###############################################################################
# checking how many purchased items in the test data that were in the 
# recommendation were purchase/viewed in the training data for collaborative
# filtering
###############################################################################
def create_groupwise_accuracy_report(recommendation_dict, view_purchase_dict):
    purchased_in_training = []
    viewed_in_training = []
    not_in_training = []
    for key, items in recommendation_dict.items():
        train_purchase = view_purchase_dict['train'][key]['purchased']
        train_viewed = view_purchase_dict['train'][key]['viewed']
        test_purchase = view_purchase_dict['test'][key]['purchased']
        for item in items:
            if item in test_purchase:
                if item in train_purchase:
                    purchased_in_training.append(item)
                elif item in train_viewed:
                    viewed_in_training.append(item)
                else:
                    not_in_training.append(item)
    return purchased_in_training, viewed_in_training, not_in_training

def items_purchased_in_testing(view_purchase_dict):
    purchased_items = []
    for key in view_purchase_dict['test']:
        for item in view_purchase_dict['test'][key]['purchased']:
            purchased_items.append(item)
    return purchased_items

view_purchase_dict = cf.item_viewed_purchased(train, test)
cf_recommendation_dict = cf.create_recommendation_for_test_users(test, rating_matrix, R_final)

purchased_items = items_purchased_in_testing(view_purchase_dict) 
purchased_in_training, viewed_in_training, not_in_training = create_groupwise_accuracy_report(cf_recommendation_dict, view_purchase_dict)
print len(not_in_training)
print len(viewed_in_training)
print len(purchased_in_training) 
               
###############################################################################
# checking how many items in the test purchase were already viewed/purchased in
# the training data
###############################################################################
def test_purchase_status_in_training_data():
    viewed = []
    purchased = []
    unseen = []
    train, test = cf.create_train_test_set(n_users = 1000, seed_value = 10)
    view_purchase_dict = cf.item_viewed_purchased(train, test)
    for key in view_purchase_dict['test']:
        if key in view_purchase_dict['train']:
            for item in view_purchase_dict['test'][key]['purchased']:
                if item in view_purchase_dict['train'][key]['purchased']:
                    purchased.append(item)
                elif item in view_purchase_dict['train'][key]['viewed']:
                    viewed.append(item)
                else:
                    unseen.append(item)
    return unseen, viewed, purchased
                    
unseen, viewed, purchased =  test_purchase_status_in_training_data()               
print len(unseen)
print len(viewed)
print len(purchased)                

per_accuracy_unseen = len(not_in_training)*100/len(unseen)
per_accuracy_viewed = len(viewed_in_training)*100/len(viewed)
per_accuracy_purchased = len(purchased_in_training)*100/len(purchased)
    
###############################################################################
# content filtering
###############################################################################
coupon_id_to_clust_id_dict, coupon_clust_def_dict = conf.get_cluster_info()
user_content_vector_dict = conf.get_user_content_vector()
coupon_content_vector_dict = conf.get_coupon_content_vector()

train, test = conf.get_train_test_set(n_users = 1000, seed_value = 10)
test_user_purchase_dict = conf.get_purchased_items_test_users(test)
train_user_purchase_dict = conf.get_purchased_items_test_users(train)
test_user_recommendation_dict = conf.get_recommendation(test, coupon_content_vector_dict, user_content_vector_dict)
conf_recommendation_dict = conf.get_recommendation(test, coupon_content_vector_dict, user_content_vector_dict)
conf.calculate_percentage_accuracy(test_user_recommendation_dict, test_user_purchase_dict)
 
###############################################################################
# checking how many purchased items in the test data that were in the 
# recommendation were purchase/viewed in the training data for content
# filtering
###############################################################################
conf_recommendation_dict =  test_user_recommendation_dict              
purchased_in_training, viewed_in_training, not_in_training = create_groupwise_accuracy_report(conf_recommendation_dict, view_purchase_dict)
print len(not_in_training)
print len(viewed_in_training)
print len(purchased_in_training) 

unseen, viewed, purchased =  test_purchase_status_in_training_data()               
print len(unseen)
print len(viewed)
print len(purchased)                

per_accuracy_unseen = len(not_in_training)*100/len(unseen)
per_accuracy_viewed = len(viewed_in_training)*100/len(viewed)
per_accuracy_purchased = len(purchased_in_training)*100/len(purchased)

###############################################################################
# how much of the purchase is covered by content filtering and collaborative 
# filtering. Are there any commanility. Are there any difference?
###############################################################################
cf_recommendation_dict
conf_recommendation_dict
purchase_dict = cf.item_purchased(train, test)

def prediction_item_comparison(cf_recommendation_dict, conf_recommendation_dict, purchase_dict):
    common = []
    only_cf = []
    only_conf = []
    none = []
    for key in purchase_dict['test']:
        for item in purchase_dict['test'][key]:
            if item in cf_recommendation_dict[key]:
                if item in conf_recommendation_dict[key]:
                    common.append(item)
                if item not in conf_recommendation_dict[key]:
                    only_cf.append(item)
            if item in conf_recommendation_dict[key]:
                if item not in cf_recommendation_dict[key]:
                    only_conf.append(item)
            if item not in cf_recommendation_dict[key]:
                if item not in conf_recommendation_dict[key]:
                    none.append(item)
    return common, only_cf, only_conf, none
                
common, only_cf, only_conf, none =  prediction_item_comparison(cf_recommendation_dict, conf_recommendation_dict, purchase_dict)             

print len(common)
print len(only_cf)
print len(only_conf)
print len(none)

###############################################################################
# checking the purchase in the 

def replace_coupon_id_with_cluster_id_in_visit_data():
    coupon_visit_train = pd.read_csv("../data/coupon_visit_train.csv")
    coupon_id_to_clust_id_dict, _ = conf.get_cluster_info()
    for key in coupon_id_to_clust_id_dict.keys():
        ind = coupon_visit_train.VIEW_COUPON_ID_hash == key
        coupon_visit_train.loc[ind, 'VIEW_COUPON_ID_hash'] = coupon_id_to_clust_id_dict[key]
    return coupon_visit_train

def get_cluster_visit_data():
    if os.path.isfile('coupon_clust_visit_train.pkl'):
        coupon_clust_visit_train = pickle.load(open('coupon_clust_visit_train.pkl','rb'))
    else:
        coupon_clust_visit_train = replace_coupon_id_with_cluster_id_in_visit_data()
        pickle.dump(coupon_clust_visit_train, open('coupon_clust_visit_train.pkl', 'wb '))
    return coupon_clust_visit_train
    
    
coupon_clust_visit_train = get_cluster_visit_data()

train_users = train.USER_ID_hash.unique()
train_users = train_users.tolist()
test_users = test.USER_ID_hash.unique()
test_users = test_users.tolist()

users = train_users + test_users
users = list(set(users))



    


      















