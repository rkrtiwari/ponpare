# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:11:49 2017

@author: ravitiwari
"""



import numpy as np
import pandas as pd

###############################################################################
# make a recommendation. send the sorted list for recommendation
###############################################################################
#1. create a ranking for a given user
coupon_content_vector_dict
user_content_vector_dict
purchase_area

###############################################################################
# view few items in the dictionary to have some idea
###############################################################################
dict(user_content_vector_dict.items()[0:2])
dict(coupon_content_vector_dict.items()[0:2])
dict(purchase_area.items()[0:2])



###############################################################################
# 1. Choose a random user
###############################################################################
def get_a_user(user_list, user_content_vector_dict, test):
    ind = pd.isnull(user_list.PREF_NAME)
    users = user_list.USER_ID_hash[~ind] 
    test_users = [user for user in users if user in test.USER_ID_hash.values]
#    users = user_content_vector_dict.keys()
    n_users = len(test_users)
    ind = np.random.choice(n_users, size = 1)
    user_id = test_users[ind]
    user_vector = user_content_vector_dict[user_id]
    return user_id, user_vector


user_id, user_content = get_a_user(user_list, user_content_vector_dict, test)   

ind = user_list.USER_ID_hash == user_id
user_content
user_list.loc[ind]


###############################################################################
# module to get a random user's coupon ranking
###############################################################################
def create_product_ranking(user_content, coupon_content_vector_dict, purchased_area):
    user_vector = user_content[0]
    user_pref = user_content[1]
    ken_area = purchased_area[user_pref]
    areas = ken_area.keys()
    
    coupon_ranking = pd.DataFrame(columns = ('coupon_id', 'ken_area',
                                         'score'))
    i = 0
    for coupon_id in coupon_content_vector_dict.keys():
        coupon_vec = coupon_content_vector_dict[coupon_id][0]
        coupon_ken = coupon_content_vector_dict[coupon_id][1]
        coupon_val = coupon_content_vector_dict[coupon_id][2]
        if coupon_ken in areas:
            score = np.dot(user_vector, coupon_vec)
            score *= ken_area[coupon_ken]/100
            coupon_ranking.loc[i] = [coupon_id, coupon_ken, score]
            i +=1
    
    coupon_ranking.sort_values(by = 'score', axis = 0, ascending = False,
                           inplace = True)
    return coupon_ranking      
            
            
product_rank = create_product_ranking(user_content, coupon_content_vector_dict, purchased_area)    
 
###############################################################################
# code for validation
###############################################################################
#1. how many coupons got the same ranking
user_id
product_rank = product_rank.drop_duplicates(subset = 'score', keep = 'first')
coupons = product_rank.coupon_id.iloc[:10]

for coupon in coupons:
    ind = coupon_list_train.COUPON_ID_hash == coupon
    print coupon_list_train[['price_rate_cat', 'price_cat', 'GENRE_NAME']].loc[ind]


#2. look at the test data
test.head()
ind = test.USER_ID_hash == user_id
test[['GENRE_NAME', 'price_cat', 'price_rate_cat']].loc[ind]




# checking why so many coupons have the same score
# finding the coupon feature of one of the coupons
coupon_ranking.iloc[0,0]
ind = coupon_list_train.COUPON_ID_hash == coupon_ranking.iloc[0,0]
genre = coupon_list_train.GENRE_NAME[ind].values[0]
discount_cat = coupon_list_train.discount_cat[ind].values[0]
price_cat = coupon_list_train.price_cat[ind].values[0]

# checking how many coupons have the same feature as the chosen coupon
ind_gen = coupon_list_train.GENRE_NAME == genre
ind_dis = coupon_list_train.discount_cat == discount_cat
ind_price = coupon_list_train.price_cat == price_cat
ind_com = ind_gen & ind_dis & ind_price
np.sum(ind_com)

coupon_list_train.loc[ind_com]
coupon_ranking.iloc[999:1005]           
