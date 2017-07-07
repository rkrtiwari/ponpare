# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 09:03:08 2017

@author: tiwarir
"""

import pandas as pd

###############################################################################
# load  coupon data
###############################################################################

def load_coupon_data():
    coupon_list_train = pd.read_csv("../data/coupon_list_train.csv")
    return coupon_list_train

def load_user_data():
    user_list = pd.read_csv("../data/user_list.csv")
    return user_list

def load_coupon_visit_data():
    coupon_visit_train = pd.read_csv("../data/coupon_visit_train.csv")
    return coupon_visit_train

