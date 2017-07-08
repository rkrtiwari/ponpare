# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 09:07:12 2017

@author: tiwarir
"""
import pandas as pd
import numpy as np
import os
import pickle
import data_loading as dl
import subset_data as sd
import random

###############################################################################
# create coupon categorical variable
###############################################################################
def create_user_categorical_variable(user_list):
    bins = [0,20,30,40,50,60,100]
    sufs = np.arange(len(bins)-1)
    labels = ["age" + str(suf) for suf in sufs]
    user_list['age_cat'] = pd.cut(user_list.AGE, bins = bins, labels = labels)
    return user_list


def create_coupon_categorical_variable(coupon_list_train):    
    #1. price rate
    bins = [-1,25,50,60,70,80,90,100]
    sufs = np.arange(len(bins)-1)
    labels = ["price_rate" + str(suf) for suf in sufs]
    coupon_list_train['price_rate_cat'] = pd.cut(coupon_list_train.PRICE_RATE, bins = bins, labels = labels)
    
    #2. catalog price
    bins = [0, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 1000000]
    sufs = np.arange(len(bins)-1)
    labels = ["catalog_price" + str(suf) for suf in sufs]
    coupon_list_train['price_cat'] = pd.cut(coupon_list_train.CATALOG_PRICE, bins = bins, labels = labels)
    return coupon_list_train



def get_coupons_cluster_id(X):
    clust_id = X.COUPON_ID_hash.values[0]
    X["coupon_cat"] = clust_id
    columns_to_keep = ['COUPON_ID_hash', 'coupon_cat']    
    return X[columns_to_keep]


def create_coupon_id_to_clust_id_dict():
    coupon_list_train = dl.load_coupon_data()
    coupon_list_train = create_coupon_categorical_variable(coupon_list_train)
    coupon_list_train = coupon_list_train.sort_values(by = 'COUPON_ID_hash')
    coupons_cluster_id = coupon_list_train.groupby(by = ['GENRE_NAME','price_cat', 'price_rate_cat']).apply(get_coupons_cluster_id)
    
    coupon_id_to_clust_id_dict = {}
    n_row, _ = coupons_cluster_id.shape
    
    for i in range(n_row):
        key = coupons_cluster_id.iat[i,0]
        value = coupons_cluster_id.iat[i,1]
        coupon_id_to_clust_id_dict[key] = value
                               
    return coupon_id_to_clust_id_dict
        
        
###############################################################################
# store the cluster information of all the coupons in a dictionary
###############################################################################
def get_coupon_id_to_cluster_id_dict():
    
    if os.path.isfile('intermediate_result/coupon_id_to_clust_id_dict.pkl'):
        coupon_id_to_clust_id_dict = pickle.load(open('intermediate_result/coupon_id_to_clust_id_dict.pkl','rb'))
    else:
        coupon_id_to_clust_id_dict = create_coupon_id_to_clust_id_dict()
        pickle.dump(coupon_id_to_clust_id_dict, open('intermediate_result/coupon_id_to_clust_id_dict.pkl', 'wb '))

    return coupon_id_to_clust_id_dict


def substitute_coupon_id_with_cluster_id(coupon_visit_data):
    df = coupon_visit_data.copy()
    coupon_id_to_clust_id_dict = get_coupon_id_to_cluster_id_dict()
    coupons_in_dict = coupon_id_to_clust_id_dict.keys()
    ind = df.VIEW_COUPON_ID_hash.isin(coupons_in_dict)
    df = df.loc[ind]    
    n = len(df)
    for i in range(n):
        coupon_id = df.VIEW_COUPON_ID_hash.iat[i]
        df.VIEW_COUPON_ID_hash.iat[i] = coupon_id_to_clust_id_dict[coupon_id]
    df = df.sort_values(by = 'PURCHASE_FLG', ascending = False) 
    columns_to_keep = ['USER_ID_hash', 'VIEW_COUPON_ID_hash', 'PURCHASE_FLG']
    return df[columns_to_keep]



###############################################################################
# get the dictionary that maps coupon cluster id to its definition 
###############################################################################
def create_coupon_clust_def_dict(coupon_list_train):
    coupon_list_train = create_coupon_categorical_variable(coupon_list_train)
    coupon_list_train = coupon_list_train.sort_values(by = ['COUPON_ID_hash'])
    cluster_info_df = coupon_list_train.drop_duplicates(subset = ['GENRE_NAME','price_cat', 'price_rate_cat'], keep = 'first')
    n, _ = cluster_info_df.shape
    coupon_clust_def_dict = {} 
    for  i in range(n):
        coupon_id =  cluster_info_df.COUPON_ID_hash.iloc[i]
        genre =  cluster_info_df.GENRE_NAME.iloc[i]
        discount =  cluster_info_df.price_rate_cat.iloc[i]
        price =  cluster_info_df.price_cat.iloc[i]
        coupon_clust_def_dict[coupon_id] = [genre, price, discount]
    return coupon_clust_def_dict


def get_coupon_clust_def_dict():
    if os.path.isfile('intermediate_result/coupon_clust_def_dict.pkl'):
        coupon_clust_def_dict = pickle.load(open('intermediate_result/coupon_clust_def_dict.pkl','rb'))
    else:
        coupon_list_train = dl.load_coupon_data()
        coupon_clust_def_dict = create_coupon_clust_def_dict(coupon_list_train)
        pickle.dump(coupon_clust_def_dict, open('intermediate_result/coupon_clust_def_dict.pkl', 'wb '))
    return coupon_clust_def_dict


        

if __name__ == '__main__':
    coupon_list_train = dl.load_coupon_data() 
    coupon_list_train = create_coupon_categorical_variable(coupon_list_train)
    coupon_clust_def_dict = get_coupon_clust_def_dict()
    coupon_id_to_cluster_id_dict = get_coupon_id_to_cluster_id_dict()
    
    n_row = coupon_list_train.shape[0]
    ind = np.random.choice(n_row)
    
    coupon_id = coupon_list_train.COUPON_ID_hash.iat[ind]
    genre = coupon_list_train.GENRE_NAME.iat[ind]
    price_cat = coupon_list_train.price_cat.iat[ind]
    price_rate_cat = coupon_list_train.price_rate_cat.iat[ind]
    
    coupon_clust_id = coupon_id_to_cluster_id_dict[coupon_id]
    coupon_clust_def = coupon_clust_def_dict[coupon_clust_id]
    
    
    print '''Test to check if a given coupon's features and the cluster it is assigned to
has the same features\n\n'''
    test_passed = True
        
    coupon_features = genre, price_cat, price_rate_cat
    print "coupon feature     cluster feature"
    for cou_feature, clust_feature in zip(coupon_features, coupon_clust_def):
        print cou_feature, clust_feature
        if cou_feature != clust_feature:
            test_passed = False
                
    if coupon_clust_id > coupon_id:
        test_passed = False
    print "____________________________________________________________________"    
    print "TEST PASSED:", test_passed
    print "____________________________________________________________________"
        

        
    print "\n"   
    print "Test to check if the substitution of coupon with coupon cluster done correctly"
    
    test_passed = True
    seed_value = random.choice(range(1000000))
    
    coupon_visit_subset = sd.create_data_subset(n_users = 1, min_purchase = 1, max_purchase = 100, seed_value = seed_value)
    coupon_clust_visit = substitute_coupon_id_with_cluster_id(coupon_visit_subset)
    print "___________________________________________________________________________________"
    print "coupon_id,   coupon_cluster_id,   assigned_cluster,  is_assignment_ok, is_order_ok"
    print "__________________________________________________________________________________"
    for ind in coupon_visit_subset.index:
        coupon_id = coupon_visit_subset.VIEW_COUPON_ID_hash.at[ind]
        if coupon_id not in coupon_id_to_cluster_id_dict.keys():
            continue
        coupon_clust_id = coupon_id_to_cluster_id_dict[coupon_id]
        assigned_cluster = coupon_clust_visit.VIEW_COUPON_ID_hash.at[ind]
        is_cluster_ok = coupon_clust_id == assigned_cluster
        if not is_cluster_ok:
            test_passed = False
        is_order_ok = coupon_clust_id <= coupon_id
        if not is_order_ok:
            test_passed = False
        print coupon_id, coupon_clust_id, assigned_cluster, is_cluster_ok, is_order_ok
        
    print "____________________________________________________________________"
    print "TEST PASSED:", test_passed
    print "____________________________________________________________________"
    



