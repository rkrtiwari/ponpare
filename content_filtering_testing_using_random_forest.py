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
new_dir = os.path.join("C:\\","USERS", "tiwarir", "Documents", "ponpare", "content_filtering_rf")
os.chdir(new_dir)
os.listdir(".")

import content_vector_using_random_forest as cf_rf
reload(cf_rf)

###############################################################################
# setting display options
###############################################################################
pd.options.display.max_columns = 100
pd.set_option('expand_frame_repr', False)

###############################################################################
# import the content filtering module
###############################################################################
# working parts
# create user content vector for all the users and store it in a dictionary
def delete_files():
    files_to_remove = ['user_content_vector_dict.pkl', 'coupon_clust_def_dict.pkl',
                   'coupon_clust_content_vector_dict.pkl', 'coupon_id_to_clust_id_dict.pkl',
                   'content_vector_using_random_forest.pyc']
    for files in files_to_remove:
        try:
            os.remove(files)
            print "removed", files
        except OSError:
            pass
       
    print "remaining files:"
    print os.listdir(".")

    
response = input("want to delete files ('y' or 'n'):")   
if response == "y":
    delete_files()
print os.listdir(".")
    
train, test = cf_rf.create_train_test_set(n_users=100)
user_content_vector_dict = cf_rf.get_user_content_vector()
coupon_clust_content_vector_dict = cf_rf.get_coupon_clust_content_vect()
coupon_id_to_cluster_id_dict = cf_rf.get_coupon_id_to_cluster_id_dict()
test_user_purchase_dict = cf_rf.get_purchased_items_test_users(test)
test_user_recommendation_dict = cf_rf.get_recommendation(test, coupon_clust_content_vector_dict, user_content_vector_dict)
print cf_rf.calculate_percentage_accuracy(test_user_recommendation_dict, test_user_purchase_dict)







   