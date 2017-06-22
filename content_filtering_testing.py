# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 13:57:56 2017

@author: tiwarir
"""
###############################################################################
# change to the appropriate working directory
###############################################################################
import os
import pandas as pd

###############################################################################
# move to appropriate directory
###############################################################################
os.getcwd()
new_dir = os.path.join("C:\\","USERS", "tiwarir", "Documents", "ponpare", "content_filtering")
os.chdir(new_dir)
os.listdir(".")

###############################################################################
# setting display options
###############################################################################
pd.options.display.max_columns = 100
pd.set_option('expand_frame_repr', False)

###############################################################################
# import the content filtering module
###############################################################################
import content_filtering_functions as conf
reload(conf)

###############################################################################
# find the recommendation and accuracy for the same set of users that was used
# in the collaborative filtering
###############################################################################
coupon_id_to_clust_id_dict, coupon_clust_def_dict = conf.get_cluster_info()
user_content_vector_dict = conf.get_user_content_vector()
coupon_content_vector_dict = conf.get_coupon_content_vector()

train, test = conf.get_train_test_set(n_users = 1000, seed_value = 10)

test_user_purchase_dict = conf.get_purchased_items_test_users(test)
test_user_recommendation_dict = conf.get_recommendation(test, coupon_content_vector_dict, user_content_vector_dict)
conf.calculate_percentage_accuracy(test_user_recommendation_dict, test_user_purchase_dict)


train_15, test_15 = conf.get_train_test_set(n_users = 100, seed_value = 15)
train_10, test_10 = conf.get_train_test_set(n_users = 100, seed_value = 10)
train_20, test_20 = conf.get_train_test_set(n_users = 100, seed_value = 20)
train_50, test_50 = conf.get_train_test_set(n_users = 100, seed_value = 50)
test_10_users = test_10.USER_ID_hash.unique()
test_15_users = test_15.USER_ID_hash.unique()
test_20_users = test_20.USER_ID_hash.unique()
test_50_users = test_50.USER_ID_hash.unique()


[user for user in test_10_users if user in test_15_users]
[user for user in test_10_users if user in test_20_users]
[user for user in test_10_users if user in test_50_users]





   