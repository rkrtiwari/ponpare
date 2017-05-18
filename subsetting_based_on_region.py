# -*- coding: utf-8 -*-
"""
Created on Tue May 09 13:39:18 2017

@author: ravitiwari
"""
from collections import defaultdict
from __future__ import division
import numpy as np
import pandas as pd
import os

###############################################################################
# change to the appropriate working directory
###############################################################################

os.getcwd()
new_dir = os.path.join("Documents", "ponpare")
os.chdir(new_dir)
###############################################################################


###############################################################################
# get data for only purchased coupons
###############################################################################

def get_purchased_coupon_data():
    fname = 'coupon_visit_train.csv'
    fname = os.path.join("data", fname)
    coupon_visit_train = pd.read_csv(fname)
    pur_ind = coupon_visit_train.PURCHASE_FLG == 1
    purchased_coupons = coupon_visit_train[pur_ind]
    columns_to_keep = ['I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash']
    purchased_coupons = purchased_coupons[columns_to_keep]
    return purchased_coupons

    
purchased_coupons =  get_purchased_coupon_data()       
purchased_coupons.head()    

##############################################################################
# merging purchased coupon data with user and coupon information
##############################################################################

def merge_purchased_coupon_to_get_location(purchased_coupons):
    fname = 'user_list.csv'
    fname = os.path.join("data", fname)
    user_list = pd.read_csv(fname)
    
    fname = 'coupon_list_train.csv'
    fname = os.path.join("data", fname)
    coupon_list_train = pd.read_csv(fname)
    
    purchased_user_info = purchased_coupons.merge(user_list, how = 'left', 
                                                     on = 'USER_ID_hash')
    purchased_user_coupon_info = purchased_user_info.merge(coupon_list_train,
                how = 'left', left_on = 'VIEW_COUPON_ID_hash', right_on = 'COUPON_ID_hash')
    columns_to_keep = ['I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash', 'PREF_NAME',
                   'ken_name']
    
    purchased_user_coupon = purchased_user_coupon_info[columns_to_keep]
    return purchased_user_coupon
    
purchased_user_coupon = merge_purchased_coupon_to_get_location(purchased_coupons)


###############################################################################
# getting the coupon purchase area count by user prefecture
###############################################################################

def get_purchased_coupon_area_by_user_area(purchased_user_coupon):
    purchased_coupon_ken_by_user_pref = pd.DataFrame(columns = ['pref', 'ken',
                                                         'count', 'per_purchase'])
    purchased_user_coupon.dropna(axis=0, how = 'any', inplace = True)
    user_prefs = purchased_user_coupon.PREF_NAME.unique()
    for pref in user_prefs:
        ind = purchased_user_coupon.PREF_NAME == pref
        df = purchased_user_coupon.loc[ind]
        df_value_counts = df.ken_name.value_counts()
        n_entry = len(df_value_counts)
        pref_s = pd.Series(index=range(n_entry))        
        for i in range(n_entry):
            pref_s.loc[i] = pref
        ken = pd.Series(df_value_counts.index)
        count = pd.Series(df_value_counts.values)
        per = count*100/np.sum(count)
        df_2 = pd.DataFrame({'pref':pref_s, 'ken': ken, 'count': count, 'per_purchase': per })
        purchased_coupon_ken_by_user_pref = pd.concat([purchased_coupon_ken_by_user_pref, df_2], ignore_index=True)
        
    purchased_coupon_ken_by_user_pref.sort_values(by = ['pref', 'count'], 
                                              inplace = True, ascending = False)  
    return purchased_coupon_ken_by_user_pref


purchased_coupon_ken_by_user_pref = get_purchased_coupon_area_by_user_area(purchased_user_coupon)        
            
purchased_coupon_ken_by_user_pref.iloc[0:5]

###############################################################################
# saving top n kens for every prefecture in a dictionary
###############################################################################

def get_top_coupon_area_for_user_area(purchased_coupon_ken_by_user_pref, n = 15):
    purchase_area = {}
    pref_names = purchased_coupon_ken_by_user_pref.pref.unique()
    for pref in pref_names:
        ind = purchased_coupon_ken_by_user_pref.pref == pref
        df = purchased_coupon_ken_by_user_pref.loc[ind]
        if df.empty:
            continue
        ken_info = {}
        for i in range(n):
            key = df.ken.values[i]
            value = df.per_purchase.values[i]
            ken_info[key] = value
            
        purchase_area[pref] = ken_info
        
    return purchase_area

purchase_area = get_top_coupon_area_for_user_area(purchased_coupon_ken_by_user_pref)


for key in purchase_area.keys():
    print key, purchase_area[key]
    

###############################################################################
# check for the algorithm validity. radomly choose a prefecture and see it gives
# the same result as the stored values
###############################################################################
def check_result(purchased_user_coupon, purchase_area):
    pref_names = purchased_user_coupon.PREF_NAME.unique()
    pref_names = purchased_user_coupon.PREF_NAME.unique()
    n_pref = len(pref_names)
    ind = np.random.choice(n_pref)
    pref = pref_names[ind]
    pref_ind = purchased_user_coupon.PREF_NAME == pref
    df = purchased_user_coupon.loc[pref_ind]
    df = df[['ken_name', 'PREF_NAME']].groupby('ken_name').agg('count')
    df = df.sort_values(by = 'PREF_NAME', ascending = False)
    df['percentage'] = df['PREF_NAME']*100/np.sum(df['PREF_NAME'])
    print df[:3]
    
    for i in range(3):
        for key in purchase_area[pref][i]:
            print key, purchase_area[pref][i][key]
        

check_result(purchased_user_coupon, purchase_area)
