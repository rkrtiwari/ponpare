# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 09:26:42 2017

@author: tiwarir
"""

from __future__ import division
import data_loading as dl
import numpy as np

###############################################################################
# creates a subset of coupon purchase/view behavior data by choosing the view/purchase
# data for only specified number of users that have n number of purchases that
# lies between min_purchase and max_purchase
###############################################################################
def create_data_subset(n_users = 100, min_purchase = 1, max_purchase = 20000, seed_value = 10):
    coupon_visit_train = dl.load_coupon_visit_data()
    ind_pur = coupon_visit_train.PURCHASE_FLG == 1
    user_ids = coupon_visit_train.loc[ind_pur].USER_ID_hash.unique()
    n_total = len(user_ids)
    np.random.seed(seed_value)
    i = 0
    users = []
    while (i < n_users):
        ind = np.random.choice(range(n_total))
        user = user_ids[ind]
        if user not in users:
            user_ind = coupon_visit_train.USER_ID_hash == user
            n_purchase = np.sum(coupon_visit_train.loc[user_ind, 'PURCHASE_FLG'])
            if (n_purchase >= min_purchase) and (n_purchase <= max_purchase):
                users.append(user)
                i+=1
    ind = coupon_visit_train.USER_ID_hash.isin(users)
    return coupon_visit_train.loc[ind]

###############################################################################
# function to test the result
###############################################################################
def test_create_data_subset(n_users, min_purchase, max_purchase, seed_value, verbose = False):
    coupon_visit_subset = create_data_subset(n_users, min_purchase, max_purchase, seed_value)
    coupon_visit_aggregated = coupon_visit_subset.groupby('USER_ID_hash')['PURCHASE_FLG'].sum()
    
    n_users_o = len(coupon_visit_subset.USER_ID_hash.unique())
    min_purchase_o = min(coupon_visit_aggregated)
    max_purchase_o = max(coupon_visit_aggregated)  
    
    print "                           Requested    Result"
    print "no of users:            ", n_users, n_users_o
    print "min number of purchase: ", min_purchase, min_purchase_o
    print "max number of purchase: ", max_purchase, max_purchase_o
    
    if verbose == True:
        print coupon_visit_aggregated

    
if __name__ == '__main__':
    n_users = 5
    min_purchase = 1
    max_purchase = 10
    seed_value = 10
    
    test_create_data_subset(n_users, min_purchase, max_purchase, seed_value, verbose = True)
    
    

