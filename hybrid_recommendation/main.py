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


###############################################################################
# chagne the working directory
###############################################################################
os.getcwd()
new_dir = os.path.join("C:\\","USERS", "tiwarir", "Documents", "ponpare", "hybrid_filtering")
os.chdir(new_dir)
os.listdir(".")
###############################################################################
# setting display options
###############################################################################
pd.options.display.max_columns = 100
pd.set_option('expand_frame_repr', False)

###############################################################################
# importing the required modules
###############################################################################
import collaborative_filtering_functions as colf
import content_filtering_functions as conf
import data_loading as dl
import data_preprocessing as dpre
import data_postprocessing as dpost
import subset_data as sd
import create_training_test_data as cttd
import popular_item_recommendations as popi
import hybrid_filtering_functions as hybf

reload(colf)
reload(conf)
reload(dl)
reload(sd)
reload(cttd)
reload(dpost)
reload(dpre)

###############################################################################
# Loading the data set
###############################################################################
coupon_visit = dl.load_coupon_visit_data()
coupon_visit_subset = sd.create_data_subset(n_users = 5000, min_purchase = 1, max_purchase = 5, seed_value = 10)
coupon_clust_visit = dpre.substitute_coupon_id_with_cluster_id(coupon_visit_subset)
train, test = cttd.create_train_test_set(coupon_clust_visit, train_frac = 0.7, seed_value = 10)

# remove files associated with old recommendation
#dpre.remove_past_results()

###############################################################################
# Getting the recommendation
###############################################################################
# collaborative filtering
#colf_users_recommendation = colf.get_recommendation_for_test_data(train, test, n_comp = 5)
colf_users_recommendation = colf.create_recommendation_for_test_data(train, test, n_comp = 5)

# content filtering, all the data has been used for training purposes
conf_users_recommendation = conf.get_recommendation_for_test_data(test) 

# popular item recommendation
popi_users_recommendation = popi.create_recommendation_based_on_popular_item(train, test)

# hybrid filtering recommendation
hybf_users_recommendation = hybf.create_recommendation_for_test_data(train, test)

###############################################################################
# result analysis: accuracy calculation
###############################################################################
view_purchase_dict = dpost.item_viewed_purchased(train, test)
per_accuracy_colf =  dpost.calculate_percentage_accuracy(colf_users_recommendation, view_purchase_dict)
per_accuracy_conf =  dpost.calculate_percentage_accuracy(conf_users_recommendation, view_purchase_dict)
per_accuracy_hybf =  dpost.calculate_percentage_accuracy(hybf_users_recommendation, view_purchase_dict)
print per_accuracy_colf
print per_accuracy_conf
print per_accuracy_hybf
dpost.plot_groupwise_purchase_accuracy(train, test, colf_users_recommendation, view_purchase_dict)
dpost.plot_groupwise_purchase_accuracy(train, test, conf_users_recommendation, view_purchase_dict)
dpost.plot_groupwise_purchase_accuracy(train, test, hybf_users_recommendation, view_purchase_dict)


###############################################################################
# input details
###############################################################################
#1. Number of users
users = list(set(test.USER_ID_hash.unique().tolist() + train.USER_ID_hash.unique().tolist()))
print "No of users: ", len(users)
#2. Number of rows
print "No of rows: ", train.shape[0] + test.shape[0]
# 3. Number of coupons
n_coupons_train = len(train.VIEW_COUPON_ID_hash.unique())
print "No of coupons: ", n_coupons_train

# 4. number of   test users
n_users_train = len(train.USER_ID_hash.unique())
n_users_test = len(test.USER_ID_hash.unique())
print "No of users in the test data: ", n_users_test
print "No of users in the test data: ", test.shape[0]

# 5. percentage of NA in rating matrix
n_ratings = train.drop_duplicates(subset = ['USER_ID_hash', 'VIEW_COUPON_ID_hash']).shape[0]
print "NA percentage in the test data: ", 1 - (n_ratings/(n_coupons_train*n_users_train))

unseen, viewed, purchased = dpost.test_purchase_status_in_training_data(train, test)
print "Unseen: ", len(unseen), "Viewed: ", len(viewed), "Purchased:", len(purchased)

###############################################################################
# Plot the results
###############################################################################
# 1. collaborative filtering
purchased_in_training, viewed_in_training, not_in_training = dpost.create_groupwise_purchase_report(colf_users_recommendation, view_purchase_dict)
print len(purchased_in_training), len(viewed_in_training), len(not_in_training)
print len(purchased_in_training)*100/len(purchased)
print len(viewed_in_training)*100/len(viewed)
print len(not_in_training)*100/len(unseen)

# 2. content filtering
purchased_in_training, viewed_in_training, not_in_training = dpost.create_groupwise_purchase_report(hybf_users_recommendation, view_purchase_dict)
print len(purchased_in_training), len(viewed_in_training), len(not_in_training)
print len(purchased_in_training)*100/len(purchased)
print len(viewed_in_training)*100/len(viewed)
print len(not_in_training)*100/len(unseen)


# 3. hybrid filtering
purchased_in_training, viewed_in_training, not_in_training = dpost.create_groupwise_purchase_report(conf_users_recommendation, view_purchase_dict)
print len(purchased_in_training), len(viewed_in_training), len(not_in_training)
print len(purchased_in_training)*100/len(purchased)
print len(viewed_in_training)*100/len(viewed)
print len(not_in_training)*100/len(unseen)

###############################################################################
# accuracy for different number of purchases
###############################################################################



