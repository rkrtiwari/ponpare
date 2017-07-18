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
new_dir = os.path.join("C:\\","USERS", "tiwarir", "Documents", "ponpare", "accuracy_anamoly_investigation")
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
import matplotlib.pyplot as plt
import numpy as np
import pickle

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
train = pickle.load(open('intermediate_result/train_user_5000_seed_10.pkl','rb'))
test = pickle.load(open('intermediate_result/test_user_5000_seed_10.pkl','rb'))

train_5 = pickle.load(open('intermediate_result/train_user_5000_seed_10_max_purchase_5.pkl','rb'))
test_5 = pickle.load(open('intermediate_result/test_user_5000_seed_10_max_purchase_5.pkl','rb'))

###############################################################################
# input details
###############################################################################
input_detail = dpost.get_input_data_details(train, test)
input_detail_5 = dpost.get_input_data_details(train_5, test_5)

###############################################################################
# Getting the recommendation
###############################################################################
colf_users_recommendation = colf.create_recommendation_for_test_data(train, test, n_comp = 5)
colf_users_recommendation_5 = colf.create_recommendation_for_test_data(train_5, test_5, n_comp = 5)

###############################################################################
# accuracy based on test coupons status in the training data i.e. if they have 
# been purchased, just viewed, or neither viewed nor purchased in the training
# data
###############################################################################
# analysis 1
accuracy_report, test_purchase_status_in_training_data, successful_recommendation_status_in_training_data = dpost.accuracy_based_on_coupon_status_in_train_data(train, test, colf_users_recommendation, view_purchase_dict)
dpost.plot_groupwise_purchase_accuracy(train, test, colf_users_recommendation, view_purchase_dict)

# analysis 2
accuracy_report_5, test_purchase_status_in_training_data_5, successful_recommendation_status_in_training_data_5 = dpost.accuracy_based_on_coupon_status_in_train_data(train_5, test_5, colf_users_recommendation_5, view_purchase_dict_5)
dpost.plot_groupwise_purchase_accuracy(train_5, test_5, colf_users_recommendation_5, view_purchase_dict_5)

###############################################################################
# accuracy for different number of purchases (PURCHASE) 
# function has been named wrongly :(
###############################################################################
df_merged = dpost.accuracy_based_on_number_of_purchases_in_test(view_purchase_dict, colf_users_recommendation, test)
dpost.plot_purchase_accuracy_for_different_number_of_users(df_merged)

df_merged_5 = dpost.accuracy_based_on_number_of_purchases_in_test(view_purchase_dict_5, colf_users_recommendation_5, test_5)
dpost.plot_purchase_accuracy_for_different_number_of_users(df_merged_5)


