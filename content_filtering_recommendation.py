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
def get_a_user(user_content_vector_dict):
    users = user_content_vector_dict.keys()
    n_users = len(users)
    ind = np.random.choice(n_users, size = 1)
    user_id = users[ind]
    user_vector = user_content_vector_dict[user_id]
    return user_id, user_vector
    
    
user_id, user_content = get_a_user(user_content_vector_dict)   


def create_product_ranking(user_content, coupon_content_vector_dict, purchase_area):
    user_vector = user_content[0]
    user_pref = user_content[1]
    ken_area = purchase_area[user_pref]
    areas = ken_area.keys()
    
    coupon_ranking = pd.DataFrame(columns = ('coupon_id', 'ken_area', 'validity',
                                         'score'))
    i = 0
    for coupon_id in coupon_content_vector_dict.keys():
        coupon_vec = coupon_content_vector_dict[coupon_id][0]
        coupon_ken = coupon_content_vector_dict[coupon_id][1]
        coupon_val = coupon_content_vector_dict[coupon_id][2]
        if coupon_ken in areas:
            score = np.dot(user_vector, coupon_vec)
            print score
            score *= ken_area[coupon_ken]/100
            print score, ken_area[coupon_ken]
            coupon_ranking.loc[i] = [coupon_id, coupon_ken, coupon_val, score]
            i +=1
    
    coupon_ranking.sort_values(by = 'score', axis = 0, ascending = False,
                           inplace = True)
    return coupon_ranking      
            
            

user_vector = user_content[0]
user_pref = user_content[1]
ken_area = purchase_area[user_pref]



product_rank = create_product_ranking(user_content, coupon_content_vector_dict, purchase_area)    
product_rank.iloc[1:50]   




user_vector = user_content[0]
user_pref = user_content[1]    
ken_area = purchase_area[user_pref]

#2. create score for all the coupons in the coupon list

i = 0
for coupon_id in coupon_content_vector_dict.keys():
    coupon_vec = coupon_content_vector_dict[coupon_id]
    score = np.dot(user_vector[0], coupon_vec[0])
    ken_area = coupon_vec[1]
    validity = coupon_vec[2]
    coupon_ranking.loc[i] = [coupon_id, ken_area, validity, score]
    i +=1
    
coupon_ranking.sort_values(by = 'score', axis = 0, ascending = False,
                           inplace = True)

#3. return the top 5 coupons in the list
coupon_ranking.iloc[0:100]  # top 5 recommended coupons


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
