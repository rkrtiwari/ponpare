# -*- coding: utf-8 -*-
"""
Created on Mon Jul 03 12:08:30 2017

@author: tiwarir
"""


###############################################################################
# import the required module
###############################################################################
from __future__ import division
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from itertools import chain
import pickle
from scipy import linalg
from numpy import dot

import collaborative_filtering_functions as cf 
reload(cf)

import content_filtering_functions as conf
reload(conf)


def item_purchased(train, test):
    purchase_dict = {}
    purchase_dict["train"] = defaultdict(lambda: [])    
    purchase_dict["test"] = defaultdict(lambda: [])
    ind_train = train.PURCHASE_FLG == 1
    ind_test = test.PURCHASE_FLG == 1
    train_purchase = train.loc[ind_train]
    test_purchase = test.loc[ind_test]
    n_train, _ = train_purchase.shape
    n_test, _ = test_purchase.shape
    for i in range(n_train):
        user = train_purchase.USER_ID_hash.iloc[i]
        coupon = train_purchase.VIEW_COUPON_ID_hash.iloc[i]
        purchase_dict["train"][user].append(coupon)
    for i in range(n_test):
        user = test_purchase.USER_ID_hash.iloc[i]
        coupon = test_purchase.VIEW_COUPON_ID_hash.iloc[i]
        purchase_dict["test"][user].append(coupon)
    return purchase_dict


def item_viewed_purchased(train, test):
    view_purchase_dict = {}
    view_purchase_dict['train'] = {}
    view_purchase_dict['test'] = {}

    train_users = train.USER_ID_hash.unique().tolist()
    test_users = test.USER_ID_hash.unique().tolist()
    
    n_train, _ = train.shape
    n_test, _ = test.shape
    
    users = list(set(train_users + test_users))
    for user in users:
        view_purchase_dict['train'][user]= {}
        view_purchase_dict['test'][user] = {}
        view_purchase_dict['train'][user]['purchased'] = []
        view_purchase_dict['train'][user]['viewed'] = []
        view_purchase_dict['test'][user]['purchased'] = []
        view_purchase_dict['test'][user]['viewed'] = []

    for i in range(n_train):
        user_id = train.USER_ID_hash.iat[i]
        coupon_id = train.VIEW_COUPON_ID_hash.iat[i]
        purchase_flg = train.PURCHASE_FLG.iat[i]
        if purchase_flg == 1:
            view_purchase_dict['train'][user_id]['purchased'].append(coupon_id)
    
    for i in range(n_train):
        user_id = train.USER_ID_hash.iat[i]
        coupon_id = train.VIEW_COUPON_ID_hash.iat[i]
        purchase_flg = train.PURCHASE_FLG.iat[i]
        if purchase_flg == 0:
            if coupon_id not in view_purchase_dict['train'][user_id]['purchased']:
                view_purchase_dict['train'][user_id]['viewed'].append(coupon_id)
                   
    for i in range(n_test):
        user_id = test.USER_ID_hash.iat[i]
        coupon_id = test.VIEW_COUPON_ID_hash.iat[i]
        purchase_flg = test.PURCHASE_FLG.iat[i]
        if purchase_flg == 1:
            view_purchase_dict['test'][user_id]['purchased'].append(coupon_id)
            
    for i in range(n_test):
        user_id = test.USER_ID_hash.iat[i]
        coupon_id = test.VIEW_COUPON_ID_hash.iat[i]
        purchase_flg = test.PURCHASE_FLG.iat[i]
        if purchase_flg == 0:
            if coupon_id not in view_purchase_dict['test'][user_id]['purchased']:
                view_purchase_dict['test'][user_id]['viewed'].append(coupon_id)
                
    return view_purchase_dict


def calculate_percentage_accuracy(recommendations_dict, purchase_dict):
    total_bought = 0
    total_correct_recommendation = 0
    for key in recommendations_dict:
        recommendation = recommendations_dict[key]
        purchase = purchase_dict['test'][key]
        total_bought += len(purchase)
        correct_recommendation = [x for x in recommendation if x in purchase]
        total_correct_recommendation += len(correct_recommendation)
    percentage_accuracy = total_correct_recommendation*100/total_bought
    return percentage_accuracy

def items_purchased_in_testing(view_purchase_dict):
    purchased_items = []
    for key in view_purchase_dict['test']:
        for item in view_purchase_dict['test'][key]['purchased']:
            purchased_items.append(item)
    return purchased_items

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

def test_purchase_status_in_training_data():
    viewed = []
    purchased = []
    unseen = []
#    train, test = cf.create_train_test_set_max_purchase_limit(n_users = 100, seed_value = 10, max_purchase_limit = 5)
    train, test = cf.create_train_test_set(n_users = 1000, seed_value = 10)
    view_purchase_dict = item_viewed_purchased(train, test)
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


