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
import numpy as np
import pickle
import content_filtering_functions as conf
from collections import defaultdict 
reload(conf)

###############################################################################
# move to appropriate directory
###############################################################################
os.getcwd()
new_dir = os.path.join("Documents", "ponpare")
os.chdir(new_dir)

###############################################################################
# setting display options
###############################################################################
pd.options.display.max_columns = 100
pd.set_option('expand_frame_repr', False)

###############################################################################
# import the content filtering module
###############################################################################
user_list, coupon_list_train = conf.load_input_files()
conf.create_user_categorical_variable(user_list)
conf.create_coupon_categorical_variable(coupon_list_train)
coupon_purchase_data =  conf.get_coupon_purchase_data(user_list, coupon_list_train)
coupon_cond_prob = conf.get_conditional_probability(coupon_purchase_data, user_list, 
                                               coupon_list_train)
user_content_vector_dict = conf.create_user_vector_dict(user_list) 
user_content_vector_dict = pickle.load(open('user_content_vector_dict.pkl','rb'))
coupon_content_vector_dict = conf.create_coupon_vector_dict(coupon_list_train, coupon_cond_prob)
coupon_content_vector_dict = pickle.load(open('coupon_content_vector_dict.pkl','rb')) 
purchase_area = conf.get_user_purchase_area(n = 3)
purchase_area1 = pickle.load(open('purchase_area_dict.pkl','rb'))

train, test = conf.create_train_test_set(n_users = 100, seed_value = 10)
train_users = train.USER_ID_hash.unique().tolist()
test_users = test.USER_ID_hash.unique().tolist()

user_info = conf.get_a_user(user_list, user_content_vector_dict, test)
recommended_coupons = conf.get_recommendation(user_info, coupon_content_vector_dict, purchase_area)







   