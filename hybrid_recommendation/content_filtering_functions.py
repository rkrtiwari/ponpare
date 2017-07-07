# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 13:52:42 2017

@author: tiwarir
"""
###############################################################################
# import the required modules
###############################################################################
from __future__ import division
import pandas as pd
import numpy as np
import os
import pickle
import data_preprocessing as dp
import data_loading as dl

reload(dp)
reload(dl)
###############################################################################
# user content vector
###############################################################################
#Function to convert age category and gender into user content  vector
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

# function to calculate user content vector of all the users of user list
# into a dictionary
def create_user_vector_dict(user_list):    
    user_list = dp.create_user_categorical_variable(user_list)
    
    user_content_vector_dict = {}
    n_users, n_features = user_list.shape
    
    for i in range(n_users):
        user_id = user_list.USER_ID_hash.iloc[i]
        gender = user_list.SEX_ID.iloc[i]
        age = user_list.age_cat.iloc[i]
        user_vector = user_vector_from_user_content(age, gender)
        user_content_vector_dict[user_id] = user_vector
    return user_content_vector_dict

def get_user_content_vector_dict():
    if os.path.isfile('user_content_vector_dict.pkl'):
        user_content_vector_dict = pickle.load(open('user_content_vector_dict.pkl','rb'))
    else:
        user_list = dl.load_user_data()
        user_content_vector_dict = create_user_vector_dict(user_list)
        pickle.dump(user_content_vector_dict, open('user_content_vector_dict.pkl', 'wb '))
    return user_content_vector_dict



###############################################################################
# coupon content vector
###############################################################################
def merge_coupon_visit_with_user_coupon_list(train):
    user_list = dl.load_user_data()
    user_list = dp.create_user_categorical_variable(user_list)
    coupon_list = dl.load_coupon_data()
    coupon_list = dp.create_coupon_categorical_variable(coupon_list)
    
    purchased_coupons = train[train.PURCHASE_FLG == 1]
    purchased_user_info = purchased_coupons.merge(user_list, how = 'left', 
                                                     on = 'USER_ID_hash')
    
    purchased_user_coupon_info = purchased_user_info.merge(coupon_list,
                how = 'left', left_on = 'VIEW_COUPON_ID_hash', right_on = 'COUPON_ID_hash')
    columns_to_keep = ['SEX_ID', 'AGE', 'age_cat','GENRE_NAME', 'PRICE_RATE',
                       'price_rate_cat', 'CATALOG_PRICE', 'price_cat']
    
    purchased_user_coupon_info = purchased_user_coupon_info[columns_to_keep]
    purchased_user_coupon_info = purchased_user_coupon_info.dropna(how = 'any')
    return purchased_user_coupon_info   
    

def get_conditional_probability(train):
    
    print "calculating conditional probability"
    
    coupon_purchase_data = merge_coupon_visit_with_user_coupon_list(train)
    
    
    c_features = ["GENRE_NAME", "price_rate_cat", "price_cat"]
    u_features = ["age_cat", "SEX_ID"]
    coupon_cond_prob = pd.DataFrame(columns = ('coupon_feature', 'user_feature', 
                                           'coupon_feature_value', 'user_feature_value',
                                           'cond_prob'))
    i = 0
    for c_feature in c_features:
        c_feature_values = coupon_purchase_data[c_feature].unique()
        c_value_count =  coupon_purchase_data[c_feature].value_counts()
        c_total = sum(c_value_count)
        for c_feature_value in c_feature_values:
            c_prob =  c_value_count.loc[c_feature_value]/c_total
            for u_feature in u_features:
                u_feature_values = coupon_purchase_data[u_feature].unique()
                u_value_count =  coupon_purchase_data[u_feature].value_counts()
                u_total = sum(u_value_count)
                
                ind = coupon_purchase_data[c_feature] == c_feature_value
                u_value_count_cond =  coupon_purchase_data[ind][u_feature].value_counts()
                u_total_cond = sum(u_value_count_cond)
                
                for u_feature_value in u_feature_values:
                    u_prob =  u_value_count.loc[u_feature_value]/u_total
                    if u_feature_value not in u_value_count_cond:
                        coupon_cond_prob.loc[i] = [c_feature, u_feature, c_feature_value,
                                    u_feature_value, 0.00000001]
                        i += 1
                        continue
                    u_prob_cond = u_value_count_cond.loc[u_feature_value]/u_total_cond
                    post_prob = c_prob*u_prob_cond/u_prob
                    coupon_cond_prob.loc[i] = [c_feature, u_feature, c_feature_value,
                                    u_feature_value, post_prob]
                    i += 1
    coupon_cond_prob.sort_values(by = ['coupon_feature', 'user_feature', 
                'coupon_feature_value', 'user_feature_value'],  inplace = True)
    return coupon_cond_prob


def create_coupon_content_vector_dict(train):
    print "creating coupon vector dictionary"    
    coupon_clust_def_dict = dp.get_coupon_clust_def_dict()
    coupon_cond_prob = get_conditional_probability(train)
    
    coupon_content_vector_dict = {}
    for key, value in coupon_clust_def_dict.items():
        genre, price, discount = value
        ind = coupon_cond_prob.coupon_feature_value == genre
        g_v = coupon_cond_prob.cond_prob.loc[ind].values
        ind = coupon_cond_prob.coupon_feature_value == price
        p_v = coupon_cond_prob.cond_prob.loc[ind].values
        ind = coupon_cond_prob.coupon_feature_value == discount
        d_v = coupon_cond_prob.cond_prob.loc[ind].values
        i_v = np.dot(g_v,p_v)
        v = np.dot(i_v,d_v)
        coupon_content_vector_dict[key] = v
    return coupon_content_vector_dict                
                
def get_coupon_content_vector_dict():
    if os.path.isfile('intermediate_result/coupon_content_vector_dict.pkl'):
        print "Exists: loading"
        coupon_content_vector_dict = pickle.load(open('intermediate_result/coupon_content_vector_dict.pkl','rb'))
    else:
        print "Does not exist: calculating"
        train = dl.load_coupon_visit_data()
        coupon_content_vector_dict = create_coupon_content_vector_dict(train)
        pickle.dump(coupon_content_vector_dict, open('intermediate_result/coupon_content_vector_dict.pkl', 'wb '))
    return coupon_content_vector_dict


###############################################################################
# make recommendation
###############################################################################
def create_distance_matrix(test, user_content_vector_dict, coupon_content_vector_dict):
    test_users = test.USER_ID_hash.unique().tolist()
    coupons = coupon_content_vector_dict.keys()
    n_row = len(test_users)
    n_col = len(coupons)
    data = np.zeros((n_row, n_col))
    distance_matrix = pd.DataFrame(data, index = test_users, columns = coupons)
    for user in test_users:
        for coupon in coupons:
            user_vector = user_content_vector_dict[user]
            coupon_vec = coupon_content_vector_dict[coupon]
            distance = np.dot(user_vector, coupon_vec)
            distance_matrix.loc[user, coupon] = distance
    return distance_matrix
    
def get_distance_matrix(test):
    if os.path.isfile('distance_matrix.pkl'):
        print "Distance Matrix Exists: loading"
        distance_matrix = pickle.load(open('intermediate_result/distance_matrix.pkl','rb'))
    else:
        print "Distance Matrix does not exist: calculating"
        user_content_vector_dict = get_user_content_vector_dict()
        coupon_content_vector_dict = get_coupon_content_vector_dict()
        distance_matrix = create_distance_matrix(test, user_content_vector_dict, coupon_content_vector_dict)
        pickle.dump(distance_matrix, open('intermediate_result/distance_matrix.pkl', 'wb '))
    return distance_matrix
    
    
def get_recommendation_for_a_user(user_id, distance_matrix):
    coupon_clust = distance_matrix.columns.tolist()
    training_users = distance_matrix.index.tolist() 
    
    if user_id not in training_users:
        return np.random.choice(coupon_clust, 10)
    
    user_ratings = distance_matrix.loc[user_id]
    user_ratings = user_ratings.sort_values(ascending = False)
    return user_ratings.index[:10].tolist()

def create_recommendation_for_test_data(test):
    test_users = test.USER_ID_hash.unique().tolist()
    test_users_recommendation = {}
    
    distance_matrix = get_distance_matrix(test)
    
    for user in test_users:
        recommendation = get_recommendation_for_a_user(user, distance_matrix)
        test_users_recommendation[user] = recommendation
                                 
    return test_users_recommendation
    

def get_recommendation_for_test_data(test):
    if os.path.isfile('conf_recommendation_dict.pkl'):
        conf_recommendation_dict = pickle.load(open('conf_recommendation_dict.pkl','rb'))
    else:
        conf_recommendation_dict = create_recommendation_for_test_data(test)
        pickle.dump(conf_recommendation_dict, open('conf_recommendation_dict.pkl', 'wb '))
   
    return conf_recommendation_dict
        






