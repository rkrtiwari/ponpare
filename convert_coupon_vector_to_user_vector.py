# -*- coding: utf-8 -*-
"""
Created on Mon May 15 08:04:13 2017

@author: ravitiwari
"""

import pandas as pd
import numpy as np
import os
from __future__ import division
from collections import defaultdict

###############################################################################
# change to the appropriate working directory
###############################################################################

os.getcwd()
new_dir = os.path.join("Documents", "ponpare")
os.chdir(new_dir)
###############################################################################


###############################################################################
# load files and create categorical variables
###############################################################################
user_list = pd.read_csv("data/user_list.csv")
coupon_list_train = pd.read_csv("data/coupon_list_train.csv")

def create_user_categorical_variable(user_list):
    bins = [0,20,30,40,50,60,100]
    sufs = np.arange(len(bins)-1)
    labels = ["age" + str(suf) for suf in sufs]
    user_list['age_cat'] = pd.cut(user_list.AGE, bins = bins, labels = labels)
    return


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
    return

create_user_categorical_variable(user_list) 
create_coupon_categorical_variable(coupon_list_train)   

user_list.head()
coupon_list_train.head()


###############################################################################
# get data for  purchased coupons
###############################################################################
def get_coupon_purchase_data(user_list, coupon_list_train):
    coupon_visit_train = pd.read_csv("data/coupon_visit_train.csv")
    purchased_coupons = coupon_visit_train[coupon_visit_train.PURCHASE_FLG == 1]

    columns_to_keep = ['I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash']
    purchased_coupons = purchased_coupons[columns_to_keep]
    
    purchased_user_info = purchased_coupons.merge(user_list, how = 'left', 
                                                     on = 'USER_ID_hash')
    
    purchased_user_coupon_info = purchased_user_info.merge(coupon_list_train,
                how = 'left', left_on = 'VIEW_COUPON_ID_hash', right_on = 'COUPON_ID_hash')
    columns_to_keep = ['SEX_ID', 'AGE', 'age_cat','GENRE_NAME', 'PRICE_RATE',
                       'price_rate_cat', 'CATALOG_PRICE', 'price_cat']
    
    purchased_user_coupon_info = purchased_user_coupon_info[columns_to_keep]
    purchased_user_coupon_info = purchased_user_coupon_info.dropna(how = 'any')
    return purchased_user_coupon_info
    
coupon_purchase_data =  get_coupon_purchase_data(user_list, coupon_list_train) 
coupon_purchase_data.head()

###############################################################################
# create categorical variables for calculation of conditional probability
# needs some modification
###############################################################################
def get_conditional_probability(coupon_purchase_data, user_list, coupon_list_train):
    c_features = ["GENRE_NAME", "price_rate_cat", "price_cat"]
    u_features = ["age_cat", "SEX_ID"]
    coupon_cond_prob = pd.DataFrame(columns = ('coupon_feature', 'user_feature', 
                                           'coupon_feature_value', 'user_feature_value',
                                           'cond_prob'))
    i = 0
    for c_feature in c_features:
        c_feature_values = coupon_list_train[c_feature].unique()
        c_value_count =  coupon_list_train[c_feature].value_counts()
        c_total = sum(c_value_count)
        for c_feature_value in c_feature_values:
            c_prob =  c_value_count.loc[c_feature_value]/c_total
            for u_feature in u_features:
                u_feature_values = user_list[u_feature].unique()
                u_value_count =  user_list[u_feature].value_counts()
                u_total = sum(u_value_count)
                
                ind = coupon_purchase_data[c_feature] == c_feature_value
                u_value_count_cond =  coupon_purchase_data[ind][u_feature].value_counts()
                u_total_cond = sum(u_value_count_cond)
                
                for u_feature_value in u_feature_values:
                    u_prob =  u_value_count.loc[u_feature_value]/u_total
                    if u_feature_value not in u_value_count_cond:
                        coupon_cond_prob.loc[i] = [c_feature, u_feature, c_feature_value,
                                    u_feature_value, 0.00000001]
                        continue
                    u_prob_cond = u_value_count_cond.loc[u_feature_value]/u_total_cond
                    post_prob = c_prob*u_prob_cond/u_prob
                    coupon_cond_prob.loc[i] = [c_feature, u_feature, c_feature_value,
                                    u_feature_value, post_prob]
                    i += 1
                    
    coupon_cond_prob.sort_values(by = ['coupon_feature', 'user_feature', 
                'coupon_feature_value', 'user_feature_value'],  inplace = True)
    return coupon_cond_prob

coupon_cond_prob = get_conditional_probability(coupon_purchase_data, user_list, 
                                               coupon_list_train)


###############################################################################
# test of coupon conditional probability
# later work. create a module out of these
###############################################################################
n = len(coupon_cond_prob)
c_ind = np.random.choice(n)
c_feature_value = coupon_cond_prob.coupon_feature_value[c_ind]
u_ind = np.random.choice(n) 
u_feature = coupon_cond_prob.user_feature[u_ind]
u_feature

ind1 = coupon_cond_prob.coupon_feature_value == c_feature_value
ind2 = coupon_cond_prob.user_feature == u_feature
ind = ind1 & ind2
coupon_cond_prob[ind]

c_feature = coupon_cond_prob.coupon_feature.loc[ind].iloc[0]
c_feature
c_feature_value

np.sum(coupon_purchase_data[c_feature] == c_feature_value)
len(coupon_purchase_data[c_feature])

coupon_prob = np.sum(coupon_purchase_data[c_feature] == c_feature_value)/len(coupon_purchase_data[c_feature])
coupon_prob

u_feat_value_count = user_list[u_feature].value_counts()
u_feat_value_count_total = np.sum(u_feat_value_count)
for u_feat in u_feat_value_count.index:
    print u_feat
    prob = u_feat_value_count.loc[u_feat]/u_feat_value_count_total
    print prob

0.162243693438*0.000654 + 0.30481353561*0.000871 + 0.273160494907*0.001749 + 0.168626765182*0.001664 + 0.0846412801119*0.004300 
###################################################################################

def user_vector_from_user_content(age_cat, sex_cat):
    a_v = np.zeros(6)
    ind = int(age_cat[3])
    a_v[ind] = 1
       
    if sex_cat == 'm':
        s_v = np.array([1,0])
    else:
        s_v = np.array([0,1])
    u_v = np.concatenate([s_v,a_v])
    return u_v


# test the code 
# a. single user    
user_vector_from_user_content( 'age1','f')    


###############################################################################
# create user vector for all the users in the user list and store it in a 
# dictionary
###############################################################################
def create_user_vector_dict(user_list):
    user_content_vector_dict = defaultdict(lambda: [[0,0,0,0,0,0,0,0],'jap'])
    n_users, n_features = user_list.shape
    
    for i in range(n_users):
        user_id = user_list.USER_ID_hash.iloc[i]
        pref = user_list.PREF_NAME.iloc[i]
        gender = user_list.SEX_ID.iloc[i]
        age = user_list.age_cat.iloc[i]
        user_vector = user_vector_from_user_content(age, gender)
        user_content_vector_dict[user_id] = [user_vector, pref]
        
    return user_content_vector_dict
   
user_content_vector_dict = create_user_vector_dict(user_list)
        
   
# testing the user_content_vector for a random user
# value in the dictionary
def check_user_content_vector():
    n_users, n_features = user_list.shape
    ind = np.random.choice(n_users, size = 1)
    user = user_list.USER_ID_hash.iloc[ind]
    print user_content_vector_dict[user.values[0]][0]
    print user_content_vector_dict[user.values[0]][1]
    ind = user_list.USER_ID_hash == user.values[0]
    print user_list.ix[ind, [1,2,4]]
    bins = [0,20,30,40,50,60,100]
    print bins

                
check_user_content_vector()

###############################################################################
# create a database of coupon vector. save it in a dictionary
###############################################################################

def coupon_feature_to_user_vector(coupon_feature_names, user_features, coupon_cond_prob = coupon_cond_prob):
    coupon_user_content = np.zeros(8)
    for cf_name, u_f in zip(coupon_feature_names,user_features):
        ind1 = coupon_cond_prob.coupon_feature_value == cf_name
        ind2 = coupon_cond_prob.user_feature == u_f
        ind = ind1 & ind2
        df = coupon_cond_prob.loc[ind]
        print cf_name, u_f
        print df
        
        if u_f == 'SEX_ID':
            u_v = df.cond_prob.values
            reversed_uv = u_v[::-1]
            fill_array = np.zeros(6)
            uv_full = np.concatenate((reversed_uv, fill_array))
            print uv_full
            coupon_user_content += uv_full
            
        else:
            u_v = df.cond_prob.values
            if len(u_v) == 6:
                fill_array = np.zeros(2)
            else:
                fill_array = np.zeros(3)            
            uv_full = np.concatenate((fill_array, u_v))
            print uv_full
            coupon_user_content += uv_full
            
    return coupon_user_content

# testing coupon_feature_to_user_vector
coupon_feature_to_user_vector(['エステ'], ['SEX_ID'], coupon_cond_prob)
c_fn = ['\xe3\x82\xb0\xe3\x83\xab\xe3\x83\xa1', 'catalog_price2', 'price_rate1']
user_features = ['SEX_ID','age_cat', 'age_cat']
coupon_cond_prob.head()

coupon_feature_to_user_vector(c_fn, user_features, coupon_cond_prob)

# get additional column in coupon list that has category information


def create_coupon_vector_dict(coupon_list_train):
    coupon_content_vector_dict = defaultdict(lambda: [[0,0,0,0,0,0,0,0], 'japan', 
                                                  ['2011-07-10', '2011-12-08']])
    n_coupons, n_features = coupon_list_train.shape
    user_features = ['SEX_ID','age_cat', 'age_cat']
    for i in xrange(2,n_coupons):
        c_id = coupon_list_train.COUPON_ID_hash.iat[i]
        genre = coupon_list_train.GENRE_NAME.iat[i]
        price = coupon_list_train.price_cat.iat[i]
        discount = coupon_list_train.price_rate_cat.iat[i]
        area = coupon_list_train.ken_name.iat[i]
        validity = [coupon_list_train.VALIDFROM.iat[i], coupon_list_train.VALIDEND.iat[i]]
        c_features = [genre, price, discount]
        c_u_vector = coupon_feature_to_user_vector(c_features, user_features)
        coupon_content_vector_dict[c_id] = [c_u_vector, area, validity]
    return coupon_content_vector_dict
        
coupon_content_vector_dict = create_coupon_vector_dict(coupon_list_train)
 

# testing the user_content_vector for a random user
# value in the dictionary

def check_coupon_content_vector():
    n_coupons, n_features = coupon_list_train.shape
    ind = np.random.choice(n_coupons, size = 1)
    coupon = coupon_list_train.COUPON_ID_hash.iloc[ind]
    print coupon_content_vector_dict[coupon.values[0]][0]
    print coupon_content_vector_dict[coupon.values[0]][1]
    print coupon_content_vector_dict[coupon.values[0]][2]
    
    ind = coupon_list_train.COUPON_ID_hash == coupon.values[0]
    print coupon_list_train.loc[ind, ['GENRE_NAME','VALIDFROM', 'VALIDEND', 'ken_name']]
    
check_coupon_content_vector()



