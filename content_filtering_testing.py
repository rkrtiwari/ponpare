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
new_dir = os.path.join("Documents", "ponpare", "content_filtering")
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

train, test = conf.create_train_test_set(n_users = 1000, seed_value = 10)

test_user_purchase_dict = conf.get_purchased_items_test_users(test)
test_user_recommendation_dict = conf.get_recommendation(test, coupon_content_vector_dict, user_content_vector_dict)
conf.calculate_percentage_accuracy(test_user_recommendation_dict, test_user_purchase_dict)









   