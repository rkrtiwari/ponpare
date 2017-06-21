# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:41:07 2017

@author: tiwarir
"""

###############################################################################
# import the required modules
###############################################################################
from __future__ import division
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor as rf
import pickle

###############################################################################
# 1. user id to user content vector
###############################################################################
def create_user_categorical_variable():
    user_list = pd.read_csv("../data/user_list.csv")
    bins = [0,20,30,40,50,60,100]
    sufs = np.arange(len(bins)-1)
    labels = ["age" + str(suf) for suf in sufs]
    user_list['age_cat'] = pd.cut(user_list.AGE, bins = bins, labels = labels)
    return user_list

def user_vector_from_user_content(age_cat, sex_cat):
    a_v = np.zeros(6)
    ind = int(age_cat[3])
    a_v[ind] = 1
       
    if sex_cat == 'f':
        s_v = np.array([1,0])
    else:
        s_v = np.array([0,1])
    u_v = np.concatenate([s_v,a_v])
    return u_v

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
# coupon id to coupon content vector
###############################################################################
# a. coupon id to coupon cluster dictionary
#1. given coupons with same value of the features, this function return the cluster
# ids of all the coupons
#1. create categorical variable
def create_coupon_categorical_variable(): 
    coupon_list_train = pd.read_csv("../data/coupon_list_train.csv")
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


def get_coupon_id_cluster(X):
    new_val = X.COUPON_ID_hash.values[0]
    X["coupon_cat"] = new_val
    columns_to_keep = ['COUPON_ID_hash', 'coupon_cat']    
    return X[columns_to_keep]

#2. given a data frame that has information about the coupon cluster, this function
# returns a dictionary that maps all the coupon ids into 
def create_coupon_id_to_cluster_dict(coupon_id_cluster_df):
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
    coupon_id_to_clust_id_dict = create_coupon_id_to_cluster_dict(coupon_id_cluster_df)
    return coupon_id_to_clust_id_dict

def get_coupon_id_to_cluster_id_dict():
    if os.path.isfile('coupon_id_to_clust_id_dict.pkl'):
        coupon_id_to_cluster_id_dict = pickle.load(open('coupon_id_to_clust_id_dict.pkl','rb'))
    else:
        coupon_id_to_cluster_id_dict = create_coupon_id_to_cluster_id_dict()
        pickle.dump(coupon_id_to_cluster_id_dict, open('coupon_id_to_clust_id_dict.pkl', 'wb '))
    return coupon_id_to_cluster_id_dict
    


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

def get_coupon_clust_def_dict():
    if os.path.isfile('coupon_clust_def_dict.pkl'):
        coupon_clust_def_dict = pickle.load(open('coupon_clust_def_dict.pkl','rb'))
    else:
        coupon_clust_def_dict = create_coupon_clust_def_dict()
        pickle.dump(coupon_clust_def_dict, open('coupon_clust_def_dict.pkl', 'wb '))
    return coupon_clust_def_dict


 

###############################################################################
# create a scoring model
###############################################################################

def get_coupon_view_purchase_data():

    user_list = create_user_categorical_variable()
    coupon_list_train = create_coupon_categorical_variable()
    
    coupon_visit_train = pd.read_csv("../data/coupon_visit_train.csv")
    ind_view = coupon_visit_train.PURCHASE_FLG == 0
    ind_buy = coupon_visit_train.PURCHASE_FLG == 1
    view_rating = np.sum(ind_buy)/np.sum(ind_view)
    
    coupon_visit_train.loc[ind_view, "PURCHASE_FLG"] = view_rating 
    
    columns_to_keep = ['VIEW_COUPON_ID_hash', 'USER_ID_hash', 'PURCHASE_FLG']
    coupons = coupon_visit_train[columns_to_keep].copy()
    
    user_info = coupons.merge(user_list, how = 'left', 
                                                     on = 'USER_ID_hash')
    
    coupon_user_info = user_info.merge(coupon_list_train,
                how = 'left', left_on = 'VIEW_COUPON_ID_hash', right_on = 'COUPON_ID_hash')
    columns_to_keep = ['SEX_ID', 'AGE', 'age_cat','GENRE_NAME', 'PRICE_RATE',
                       'price_rate_cat', 'CATALOG_PRICE', 'price_cat', 'PURCHASE_FLG']
    
    coupon_user_info = coupon_user_info[columns_to_keep]
    coupon_user_info = coupon_user_info.dropna(how = 'any')
    return coupon_user_info 



# create dictionary for sex id encoding
def create_sex_id_encoding_dict():
    user_list = pd.read_csv("../data/user_list.csv")
    sex_id = user_list.SEX_ID.unique()
    sex_id = np.sort(sex_id).tolist()
    sex_id_dict = {}
    for i, gender in enumerate(sex_id):
        sex_id_dict[gender] = i
    return sex_id_dict    
    
    
    
def create_genre_encoding_dict():
    coupon_list_train = pd.read_csv("../data/coupon_list_train.csv")
    genres = coupon_list_train.GENRE_NAME.unique()
    genres = np.sort(genres).tolist()
    genres_dict = {}
    for i, genre in enumerate(genres):
        genres_dict[genre] = i
    return genres_dict


        
def create_age_encoding_dict():
    user_list = create_user_categorical_variable()
    age_cat = user_list.age_cat.unique()
    age_cat = np.sort(age_cat).tolist()
    age_cat_dict = {}
    for i, age in enumerate(age_cat):
        age_cat_dict[age] = i
    return age_cat_dict
        


def create_price_encoding_dict():
    coupon_list_train = create_coupon_categorical_variable()
    price_cat = coupon_list_train.price_cat.unique()
    price_cat = np.sort(price_cat).tolist()
    price_dict = {}
    for i in range(11):
        key = 'catalog_price' + str(i)
        value = i
        price_dict[key] = value
    return price_dict
        


def create_discount_encoding_dict():
    coupon_list_train = create_coupon_categorical_variable()
    discount_cat = coupon_list_train.price_rate_cat.unique()
    discount_cat = np.sort(discount_cat).tolist()
    discount_dict = {}
    for i, discount in enumerate(discount_cat):
        discount_dict[discount] = i
    return discount_dict 
    
  

def get_coupon_view_purchase_encoded_df():
    sex_id_encoding_dict = create_sex_id_encoding_dict()
    genre_encoding_dict = create_genre_encoding_dict()
    age_cat_encoding_dict = create_age_encoding_dict()
    price_encoding_dict = create_price_encoding_dict()
    discount_encoding_dict = create_discount_encoding_dict()
    
    coupon_view_purchase_df = get_coupon_view_purchase_data()
    coupon_view_purchase_encoded = coupon_view_purchase_df.copy()
    columns_to_keep = ['SEX_ID', 'age_cat', 'GENRE_NAME', 'price_rate_cat', 'price_cat', 'PURCHASE_FLG']
    coupon_view_purchase_encoded = coupon_view_purchase_encoded[columns_to_keep]
    
    # replace categorical values with their encoded values
    for key, value in sex_id_encoding_dict.items():
        ind = coupon_view_purchase_encoded.SEX_ID == key
        coupon_view_purchase_encoded.loc[ind, "SEX_ID"] = value
                                        
    for key, value in genre_encoding_dict.items():
        ind = coupon_view_purchase_encoded.GENRE_NAME == key
        coupon_view_purchase_encoded.loc[ind, "GENRE_NAME"] = value
                                        
    for key, value in age_cat_encoding_dict.items():
        ind = coupon_view_purchase_encoded.age_cat == key
        coupon_view_purchase_encoded.loc[ind, "age_cat"] = value
        
    for key, value in price_encoding_dict.items():
        ind = coupon_view_purchase_encoded.price_cat == key
        coupon_view_purchase_encoded.loc[ind, "price_cat"] = value  
                                    
    for key, value in discount_encoding_dict.items():
        ind = coupon_view_purchase_encoded.price_rate_cat == key
        coupon_view_purchase_encoded.loc[ind, "price_rate_cat"] = value  

    return coupon_view_purchase_encoded

        
###############################################################################
# create a coupon content vector
###############################################################################    
def create_coupon_clust_content_vect_dict():
    coupon_view_purchase_encoded = get_coupon_view_purchase_encoded_df()
    X_gender = coupon_view_purchase_encoded.iloc[:,[0,2,3,4]].values
    X_age = coupon_view_purchase_encoded.iloc[:, [1,2,3,4]].values
    y = coupon_view_purchase_encoded.iloc[:, 5].values
                                         
    clf_gender = rf(n_estimators=25)
    clf_gender.fit(X_gender, y)
    
    clf_age = rf(n_estimators=25)
    clf_age.fit(X_age, y)
    
    genre_encoding_dict = create_genre_encoding_dict()
    discount_encoding_dict = create_discount_encoding_dict()
    price_encoding_dict = create_price_encoding_dict()
    
    coupon_clust_content_vector = {}
    coupon_clust_def_dict = get_coupon_clust_def_dict()
    for key, value in coupon_clust_def_dict.items():
        genre, price, discount = value
        genre_code = genre_encoding_dict[genre]
        price_code = price_encoding_dict[price]
        discount_code = discount_encoding_dict[discount]
        content_vect = []
        for i in range(2):
            X = np.array([i, genre_code, discount_code, price_code])
            val = clf_gender.predict(X.reshape(1,-1))
            content_vect.append(val[0])
        for i in range(6):
            X = np.array([i, genre_code, discount_code, price_code])
            val = clf_age.predict(X.reshape(1,-1))
            content_vect.append(val[0])
        coupon_clust_content_vector[key] = content_vect
    return coupon_clust_content_vector
            
def get_coupon_clust_content_vect():
    if os.path.isfile('coupon_clust_content_vector_dict.pkl'):
        coupon_clust_content_vector_dict = pickle.load(open('coupon_clust_content_vector_dict.pkl','rb'))
    else:
        coupon_clust_content_vector_dict = create_coupon_clust_content_vect_dict()
        pickle.dump(coupon_clust_content_vector_dict, open('coupon_clust_content_vector_dict.pkl', 'wb '))
    return coupon_clust_content_vector_dict

###############################################################################
# get the training and test data. this implemenation multiple visit information
# is gone. Need to refine so that multiple visit information is  preserved
###############################################################################
def get_users_with_at_least_one_purchase(n=100, seed_value = 10):
    coupon_visit_train = pd.read_csv("../data/coupon_visit_train.csv")
    ind_pur = coupon_visit_train.PURCHASE_FLG == 1
    user_ids = coupon_visit_train.loc[ind_pur].USER_ID_hash.unique()
    n_users = len(user_ids)
    np.random.seed(seed_value)
    ind = np.random.choice(range(n_users), size = n, replace = False)
    return user_ids[ind]

def get_visit_data_for_users_with_purchase(users_with_purchase):
    coupon_visit_train = pd.read_csv("../data/coupon_visit_train.csv")
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
    columns_to_keep = ['USER_ID_hash', 'VIEW_COUPON_ID_hash', 'PURCHASE_FLG']
    return coupon_visit_selected_users[columns_to_keep]


# standard function to create training and testing data set
def create_train_test_set(n_users = 100, seed_value = 10):
    fname_train = "train_" + "user_"+ str(n_users) + "_seed_" + str(seed_value) + ".pkl"
    fname_test = "test_" + "user_"+ str(n_users) + "_seed_" + str(seed_value) + ".pkl"
    if os.path.isfile(fname_train):
        train = pickle.load(open(fname_train,'rb'))
        test = pickle.load(open(fname_test,'rb'))
    else:
        users_with_purchase = get_users_with_at_least_one_purchase(n_users, seed_value)
        coupon_visit_selected_users = get_visit_data_for_users_with_purchase(users_with_purchase)   
        coupon_visit_selected_users = substitute_coupon_id_with_cluster_id(coupon_visit_selected_users)
        np.random.seed(100)
        n_obs = len(coupon_visit_selected_users)
        ind_train = np.random.choice(n_obs, size = int(0.7*n_obs), replace = False)
        ind_test = [x for x in range(n_obs) if x not in ind_train]
        train = coupon_visit_selected_users.iloc[ind_train]
        test = coupon_visit_selected_users.iloc[ind_test]
        pickle.dump(train, open(fname_train, 'wb '))
        pickle.dump(test, open(fname_test, 'wb '))
    return train, test
        
        
        
###############################################################################
# get items purchase during testing
###############################################################################    
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



    
###############################################################################
# get the coupon purchase data with user and coupon information
###############################################################################
if __name__ == "__main__":
    user_content_vector_dict = get_user_content_vector()
    coupon_clust_content_vector_dict = get_coupon_clust_content_vect()
    coupon_id_to_cluster_id_dict = get_coupon_id_to_cluster_id_dict()
    train, test = create_train_test_set(n_users=100)
    test_user_purchase_dict = get_purchased_items_test_users(test)
    test_user_recommendation_dict = get_recommendation(test, coupon_clust_content_vector_dict, user_content_vector_dict)
    print calculate_percentage_accuracy(test_user_recommendation_dict, test_user_purchase_dict)






