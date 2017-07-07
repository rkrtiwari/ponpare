# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 14:13:25 2017

@author: tiwarir
"""

def create_recommendation_based_on_popular_item(train, test):
    test_users = test.USER_ID_hash.tolist()
    test_users_recommendation = {}
    
    top_items = train.groupby(by = 'VIEW_COUPON_ID_hash', as_index = False).sum()
    top_items = top_items.sort_values(by = 'PURCHASE_FLG', ascending = False)['VIEW_COUPON_ID_hash'][:10]
    recommendation = top_items.tolist()
    for user in test_users:
        test_users_recommendation[user] = recommendation
    return test_users_recommendation

