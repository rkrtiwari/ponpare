# -*- coding: utf-8 -*-
"""
Created on Mon May 15 08:04:13 2017

@author: ravitiwari
"""

import pandas as pd

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







###############################################################################
# create a database of user vectors. save it in a dictionary
###############################################################################

#1. function to create user vector from user content

import numpy as np
import pandas as pd

def convert_user_features_into_categories(user_list):
    bins = [0,20,30,40,50,60,100]
    sufs = np.arange(len(bins)-1)
    labels = ["age" + str(suf) for suf in sufs]
    user_list['age_cat'] = pd.cut(user_list.AGE, bins = bins, labels = labels)    
    return

def convert_coupon_features_into_categories(coupon_list_train):
    bins = [-1,25,50,60,70,80,90,100]
    sufs = np.arange(len(bins)-1)
    labels = ["price_rate" + str(suf) for suf in sufs]
    coupon_list_train['discount_cat'] = pd.cut(coupon_list_train.PRICE_RATE, bins = bins, labels = labels)
    
    bins = [0, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 1000000]
    sufs = np.arange(len(bins)-1)
    labels = ["catalog_price" + str(suf) for suf in sufs]
    coupon_list_train['price_cat'] = pd.cut(coupon_list_train.CATALOG_PRICE, bins = bins, labels = labels)
    
    return


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


# create user vector for all the users in the user list and store it in a dictionary
from collections import defaultdict
user_content_vector_dict = defaultdict(lambda: [[0,0,0,0,0,0,0,0],'jap'])

# get additional column in the user list that has category information
convert_user_features_into_categories(user_list)

n_users, n_features = user_list.shape
for i in xrange(n_users):
    user_id = user_list.USER_ID_hash.iat[i]
    pref = user_list.PREF_NAME.iat[i]
    gender = user_list.SEX_ID.iat[i]
    age = user_list.age_cat.iat[i]
       
    user_vector = user_vector_from_user_content(age, gender)
    user_content_vector_dict[user_id] = [user_vector, pref]
    
# testing the user_content_vector for a random user
# value in the dictionary
ind = np.random.choice(n_users, size = 1)
user = user_list.USER_ID_hash.iloc[ind]
print user_content_vector_dict[user.values[0]]

# value in the user_list
ind = user_list.USER_ID_hash == user.values[0]
user_list.loc[ind]

###############################################################################
# create a database of coupon vector. save it in a dictionary
###############################################################################

def coupon_feature_to_user_vector(coupon_feature_names, user_features, coupon_cond_prob = coupon_cond_prob):
    coupon_user_content = np.zeros(8)
    w = [1,1,1]
    for cf_name, u_f in zip(coupon_feature_names,user_features):
        ind1 = coupon_cond_prob.coupon_feature_value == cf_name
        ind2 = coupon_cond_prob.user_feature == u_f
        ind = ind1 & ind2
        df = coupon_cond_prob.loc[ind]
        
        if u_f == 'SEX_ID':
            u_v = df.cond_prob.values
            fill_array = np.zeros(6)
            uv_full = np.concatenate((u_v, fill_array))
            coupon_user_content += w[0]*uv_full
            
        else:
            u_v = df.cond_prob.values
            fill_array = np.zeros(2)
            uv_full = np.concatenate((fill_array, u_v))
            coupon_user_content += uv_full
            
    return coupon_user_content


# get additional column in coupon list that has category information
convert_coupon_features_into_categories(coupon_list_train)
user_features = ['SEX_ID','AGE', 'AGE']

# adding additional information in the coupon
from collections import defaultdict
coupon_content_vector_dict = defaultdict(lambda: [[0,0,0,0,0,0,0,0], 'japan', 
                                                  ['2011-07-10', '2011-12-08']])
n_coupons, n_features = coupon_list_train.shape    
for i in xrange(n_coupons):
    c_id = coupon_list_train.COUPON_ID_hash.iat[i]
    genre = coupon_list_train.GENRE_NAME.iat[i]
    price = coupon_list_train.price_cat.iat[i]    
    discount = coupon_list_train.discount_cat.iat[i]
    area = coupon_list_train.ken_name.iat[i]
    validity = [coupon_list_train.VALIDFROM.iat[i], coupon_list_train.VALIDEND.iat[i] ]
    c_features = [genre, price, discount]
    c_u_vector = coupon_feature_to_user_vector(c_features, user_features)
    coupon_content_vector_dict[c_id] = [c_u_vector, area, validity]

    
# testing the user_content_vector for a random user
# value in the dictionary
ind = np.random.choice(n_coupons, size = 1)
coupon = coupon_list_train.COUPON_ID_hash.iloc[ind]
print coupon_content_vector_dict[coupon.values[0]]

# value in the user_list
ind = coupon_list_train.COUPON_ID_hash == coupon.values[0]
coupon_list_train.loc[ind]


###############################################################################
# make a recommendation. send the sorted list for recommendation
###############################################################################
#1. create a ranking for a given user
coupon_content_vector_dict
user_content_vector_dict

#1. choosing a random user
ind = np.random.choice(n_users, size = 1)
user = user_list.USER_ID_hash.iloc[ind]
user_id = user.values[0]

user_vector = user_content_vector_dict[user_id]
user_vector[0]

#2. create score for all the coupons in the coupon list
coupon_ranking = pd.DataFrame(columns = ('coupon_id', 'ken_area', 'validity',
                                         'score'))
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
coupon_ranking.iloc[0:5,0]  # top 5 recommended coupons


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

