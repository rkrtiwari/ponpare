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

###############################################################################
# create user content vector dictionary
###############################################################################
#1. function to create a age categorical varible in the user list
def create_user_categorical_variable():
    user_list = pd.read_csv("data/user_list.csv")
    bins = [0,20,30,40,50,60,100]
    sufs = np.arange(len(bins)-1)
    labels = ["age" + str(suf) for suf in sufs]
    user_list['age_cat'] = pd.cut(user_list.AGE, bins = bins, labels = labels)
    return user_list

#2. Function to convert age category and gender into user content  vector
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


# 3. function to calculate user content vector of all the users of user list
# into a dictionary

def create_user_vector_dict():
    
    user_list = create_user_categorical_variable()
    
    user_content_vector_dict = {}
    n_users, n_features = user_list.shape
    
    for i in range(n_users):
        user_id = user_list.USER_ID_hash.iloc[i]
        gender = user_list.SEX_ID.iloc[i]
        age = user_list.age_cat.iloc[i]
        user_vector = user_vector_from_user_content(age, gender)
        user_content_vector_dict[user_id] = user_vector
    return user_content_vector_dict

def get_user_content_vector():
    if os.path.isfile('user_content_vector_dict.pkl'):
        user_content_vector_dict = pickle.load(open('user_content_vector_dict.pkl','rb'))
    else:
        user_content_vector_dict = create_user_vector_dict()
        pickle.dump(user_content_vector_dict, open('user_content_vector_dict.pkl', 'wb '))
    return user_content_vector_dict




###############################################################################
# create coupon content vector dictionary
###############################################################################
#1. create categorical variable
def create_coupon_categorical_variable(): 
    coupon_list_train = pd.read_csv("data/coupon_list_train.csv")
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

#2. subset data to get only the purchased coupon info 
def get_data_for_purchased_coupon():
    coupon_visit_train = pd.read_csv("data/coupon_visit_train.csv")
    purchased_coupons = coupon_visit_train[coupon_visit_train.PURCHASE_FLG == 1]

    columns_to_keep = ['I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash']
    purchased_coupons = purchased_coupons[columns_to_keep]    
    return purchased_coupons

#3. merge the purchase coupon data with user information and coupon information
def get_user_n_coupon_info_for_purchased_coupons():
    coupon_list_train = create_coupon_categorical_variable()
    user_list = create_user_categorical_variable()    
    purchased_coupons = get_data_for_purchased_coupon()
        
    purchased_user_info = purchased_coupons.merge(user_list, how = 'left', 
                                                     on = 'USER_ID_hash')
    
    purchased_user_coupon_info = purchased_user_info.merge(coupon_list_train,
                how = 'left', left_on = 'VIEW_COUPON_ID_hash', right_on = 'COUPON_ID_hash')
    columns_to_keep = ['SEX_ID', 'AGE', 'age_cat','GENRE_NAME', 'PRICE_RATE',
                       'price_rate_cat', 'CATALOG_PRICE', 'price_cat']
    
    purchased_user_coupon_info = purchased_user_coupon_info[columns_to_keep]
    purchased_user_coupon_info = purchased_user_coupon_info.dropna(how = 'any')
    return purchased_user_coupon_info

###############################################################################
# create categorical variables for calculation of conditional probability
# needs some modification
###############################################################################
def get_conditional_probability():
    print "using conditional probability 1"
    
    coupon_purchase_data = get_user_n_coupon_info_for_purchased_coupons()
    user_list = create_user_categorical_variable()
    coupon_list_train = create_coupon_categorical_variable()
    
    
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

def get_conditional_probability2():
    print "using conditional probability 2"
    
    coupon_purchase_data = get_user_n_coupon_info_for_purchased_coupons()
    
    
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
                        continue
                    u_prob_cond = u_value_count_cond.loc[u_feature_value]/u_total_cond
                    post_prob = c_prob*u_prob_cond/u_prob
                    coupon_cond_prob.loc[i] = [c_feature, u_feature, c_feature_value,
                                    u_feature_value, post_prob]
                    i += 1
    coupon_cond_prob.sort_values(by = ['coupon_feature', 'user_feature', 
                'coupon_feature_value', 'user_feature_value'],  inplace = True)
    return coupon_cond_prob

#
#c_features = ["GENRE_NAME", "price_rate_cat", "price_cat"]
#u_features = ["age_cat", "SEX_ID"]
#coupon_purchase_data = get_user_n_coupon_info_for_purchased_coupons()
#c_feature_value = coupon_purchase_data.GENRE_NAME.unique().tolist()
#u_feature_value = coupon_purchase_data.SEX_ID.unique().tolist()
#
#coupon_purchase_data.GENRE_NAME.value_counts()
#coupon_purchase_data.SEX_ID.value_counts()
#u_feature_value



def coupon_feature_to_user_vector(coupon_feature_names, user_features, coupon_cond_prob):
    coupon_user_content = np.zeros(8)
    for cf_name, u_f in zip(coupon_feature_names,user_features):
        ind1 = coupon_cond_prob.coupon_feature_value == cf_name
        ind2 = coupon_cond_prob.user_feature == u_f
        ind = ind1 & ind2
        df = coupon_cond_prob.loc[ind]
        
        if u_f == 'SEX_ID':
            u_v = df.cond_prob.values
            fill_array = np.zeros(8-len(u_v))
            reversed_uv = u_v[::-1]            
            uv_full = np.concatenate((reversed_uv, fill_array))
            coupon_user_content += uv_full
            
        else:
            u_v = df.cond_prob.values 
            fill_array = np.zeros(8-len(u_v))
            uv_full = np.concatenate((fill_array, u_v))
            coupon_user_content += uv_full            
    return coupon_user_content

def create_coupon_vector_dict():
    coupon_list_train = create_coupon_categorical_variable()
    coupon_cond_prob = get_conditional_probability2()
    coupon_id_to_clust_id_dict, _ = get_cluster_info()
    
    n, _ = coupon_list_train.shape
    for i in range(n):
        coupon_id = coupon_list_train.COUPON_ID_hash.iloc[i]
        cluster_id = coupon_id_to_clust_id_dict[coupon_id]
        col_ind = coupon_list_train.columns.get_loc("COUPON_ID_hash")
        coupon_list_train.iloc[i, col_ind] = cluster_id
    coupon_list_train = coupon_list_train.drop_duplicates(subset = 'COUPON_ID_hash')       
    coupon_content_vector_dict = {}
    n_coupons, _ = coupon_list_train.shape
    user_features = ['SEX_ID','age_cat', 'age_cat']
    print n_coupons
    for i in xrange(n_coupons):
        c_id = coupon_list_train.COUPON_ID_hash.iat[i]
        genre = coupon_list_train.GENRE_NAME.iat[i]
        price = coupon_list_train.price_cat.iat[i]
        discount = coupon_list_train.price_rate_cat.iat[i]
        c_features = [genre, price, discount]
        c_u_vector = coupon_feature_to_user_vector(c_features, user_features, coupon_cond_prob)
        coupon_content_vector_dict[c_id] = c_u_vector
    return coupon_content_vector_dict

def get_coupon_content_vector():
    if os.path.isfile('coupon_content_vector_dict.pkl'):
        print "Exists: loading"
        coupon_content_vector_dict = pickle.load(open('coupon_content_vector_dict.pkl','rb'))
    else:
        print "Does not exist: calculating"
        coupon_content_vector_dict = create_coupon_vector_dict()
        pickle.dump(coupon_content_vector_dict, open('coupon_content_vector_dict.pkl', 'wb '))
    return coupon_content_vector_dict

    
###############################################################################
# get the list of coupon groups and name for the group
###############################################################################
#1. given a coupon with same value of the features, this function return the cluster
# ids of all the coupons
def get_coupon_id_cluster(X):
    new_val = X.COUPON_ID_hash.values[0]
    X["coupon_cat"] = new_val
    columns_to_keep = ['COUPON_ID_hash', 'coupon_cat']    
    return X[columns_to_keep]

#2. given a data frame that has information about the coupon cluster, this function
# returns a dictionary that maps all the coupon ids into 
def get_coupon_id_to_cluster_dict(coupon_id_cluster_df):
    coupon_id_to_clust_dict = {}
    n_row, _ = coupon_id_cluster_df.shape
    for i in xrange(n_row):
        key = coupon_id_cluster_df.iat[i,0]
        value = coupon_id_cluster_df.iat[i,1]
        coupon_id_to_clust_dict[key] = value
    return coupon_id_to_clust_dict

#3. use the above two functons to create a dictionary that maps coupons ids to 
# cluster ids
def create_coupon_id_to_cluster_id_dict():
    coupon_list_train = create_coupon_categorical_variable()
    coupon_list_train = coupon_list_train.sort_values(by = 'COUPON_ID_hash')
    coupon_id_cluster_df = coupon_list_train.groupby(by = ['GENRE_NAME','price_cat', 'price_rate_cat']).apply(get_coupon_id_cluster)
    coupon_id_to_clust_id_dict = get_coupon_id_to_cluster_dict(coupon_id_cluster_df)
    return coupon_id_to_clust_id_dict
    

# 4. save the feature of all the coupon clusters in a dictionary
def create_coupon_clust_def_dict():
    coupon_list_train = create_coupon_categorical_variable()
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
        


###############################################################################
# get the dictionary that maps coupon id to coupon cluster and coupon cluster to 
# its features
###############################################################################
def get_cluster_info():
    if os.path.isfile('coupon_clust_def_dict.pkl'):
        coupon_clust_def_dict = pickle.load(open('coupon_clust_def_dict.pkl','rb'))
    else:
        coupon_clust_def_dict = create_coupon_clust_def_dict()
        pickle.dump(coupon_clust_def_dict, open('coupon_clust_def_dict.pkl', 'wb '))
        
    if os.path.isfile('coupon_id_to_clust_id_dict.pkl'):
        coupon_id_to_clust_id_dict = pickle.load(open('coupon_id_to_clust_id_dict.pkl','rb'))
    else:
        coupon_id_to_clust_id_dict = create_coupon_id_to_cluster_id_dict()
        pickle.dump(coupon_id_to_clust_id_dict, open('coupon_id_to_clust_id_dict.pkl', 'wb '))
    return coupon_id_to_clust_id_dict, coupon_clust_def_dict



###############################################################################
# selecting an user and creating its recommendation
###############################################################################
def get_users_with_at_least_one_purchase(n=100, seed_value = 10):
    coupon_visit_train = pd.read_csv("data/coupon_visit_train.csv")
    ind_pur = coupon_visit_train.PURCHASE_FLG == 1
    user_ids = coupon_visit_train.loc[ind_pur].USER_ID_hash.unique()
    n_users = len(user_ids)
    np.random.seed(seed_value)
    ind = np.random.choice(range(n_users), size = n, replace = False)
    return user_ids[ind]

def get_visit_data_for_users_with_purchase(users_with_purchase):
    coupon_visit_train = pd.read_csv("data/coupon_visit_train.csv")
    columns_to_keep = ['PURCHASE_FLG', 'VIEW_COUPON_ID_hash', 'USER_ID_hash']
    ind = coupon_visit_train.USER_ID_hash.isin(users_with_purchase)
    coupon_visit_select_users = coupon_visit_train[columns_to_keep].loc[ind]   
    return coupon_visit_select_users

def substitute_coupon_id_with_cluster_id(coupon_visit_selected_users):
    coupon_id_to_clust_dict = pickle.load(open('coupon_id_to_clust_id_dict.pkl','rb'))
    coupons_in_dict = coupon_id_to_clust_dict.keys()
    ind = coupon_visit_selected_users.VIEW_COUPON_ID_hash.isin(coupons_in_dict)
    coupon_visit_selected_users = coupon_visit_selected_users.loc[ind]    
    n = len(coupon_visit_selected_users)
    for i in range(n):
        coupon_id = coupon_visit_selected_users.VIEW_COUPON_ID_hash.iat[i]
        coupon_visit_selected_users.VIEW_COUPON_ID_hash.iat[i] = coupon_id_to_clust_dict[coupon_id]
    coupon_visit_selected_users = coupon_visit_selected_users.sort_values(by = 'PURCHASE_FLG', ascending = False)
 #   coupon_visit_selected_users = coupon_visit_selected_users.drop_duplicates(subset = 
 #   ['USER_ID_hash', 'VIEW_COUPON_ID_hash'], keep = 'first') 
    columns_to_keep = ['USER_ID_hash', 'VIEW_COUPON_ID_hash', 'PURCHASE_FLG']
    return coupon_visit_selected_users[columns_to_keep]


def create_train_test_set(n_users = 100, seed_value = 10):
    users_with_purchase = get_users_with_at_least_one_purchase(n_users, seed_value)
    coupon_visit_selected_users = get_visit_data_for_users_with_purchase(users_with_purchase)   
    coupon_visit_selected_users = substitute_coupon_id_with_cluster_id(coupon_visit_selected_users)    
    n_obs = len(coupon_visit_selected_users)
    np.random.seed(100)
#    np.random.seed(seed_value)
    ind_train = np.random.choice(n_obs, size = int(0.7*n_obs), replace = False)
    ind_test = [x for x in range(n_obs) if x not in ind_train]
    train = coupon_visit_selected_users.iloc[ind_train]
    test = coupon_visit_selected_users.iloc[ind_test]
    return train, test 

def get_train_test_set(n_users = 100, seed_value = 10):
    fname =  "user_"+ str(n_users) + "_seed_" + str(seed_value) + ".pkl"
    if os.path.isfile(fname):
        train, test = pickle.load(open(fname,'rb'))
    else:
        train, test = create_train_test_set(n_users = n_users, seed_value = seed_value)
        pickle.dump((train, test), open(fname, 'wb '))
    return train, test
        
    

def get_purchased_items_test_users(test):
    ind_pur = test.PURCHASE_FLG == 1
    test1 = test.loc[ind_pur]
    test1 = test1.drop_duplicates(subset = ['USER_ID_hash', 'VIEW_COUPON_ID_hash'], keep = 'first')
    n, _ = test1.shape
    test_user_purchase_dict = {user_id : [] for  user_id in test1.USER_ID_hash}
    for i in range(n):
        user_id = test1.USER_ID_hash.iloc[i]
        coupon = test1.VIEW_COUPON_ID_hash.iloc[i]
        test_user_purchase_dict[user_id].append(coupon)
    return test_user_purchase_dict 
        

def create_product_ranking(user_vector, coupon_content_vector_dict):
    coupon_ranking = pd.DataFrame(columns = ('coupon_id',
                                         'score'))
    i = 0
    for coupon_id in coupon_content_vector_dict.keys():
        coupon_vec = coupon_content_vector_dict[coupon_id]
        score = np.dot(user_vector, coupon_vec)
        coupon_ranking.loc[i] = [coupon_id, score]
        i += 1

    coupon_ranking.sort_values(by = 'score', axis = 0, ascending = False,
                           inplace = True)
    return coupon_ranking      


def get_recommendation(test, coupon_content_vector_dict, user_content_vector_dict):
    user_ids = test.USER_ID_hash.unique()
    n = len(user_ids)
    test_user_recommendation = {}    
#    test_user_recommendation = {k: [] for k in user_ids}
    for i in range(n):
        user_id = user_ids[i]
        user_vector = user_content_vector_dict[user_id]
        product_rank = create_product_ranking(user_vector, coupon_content_vector_dict)
        coupons = product_rank.coupon_id.values[:10]
        test_user_recommendation[user_id] = coupons   
    return test_user_recommendation
        

###############################################################################
# find out percentage accuracy
###############################################################################
def calculate_percentage_accuracy(recommendations_dict, purchase_dict):
    total_bought = 0
    total_correct_recommendation = 0
    for key in purchase_dict:
        recommendation = recommendations_dict[key]
        purchase = purchase_dict[key]
        total_bought += len(purchase)
        correct_recommendation = [x for x in recommendation if x in purchase]
        total_correct_recommendation += len(correct_recommendation)
    percentage_accuracy = total_correct_recommendation*100/total_bought
    return percentage_accuracy


        

#if __name__ == "__main__":  




