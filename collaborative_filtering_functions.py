# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 10:47:35 2017

@author: tiwarir
"""
###############################################################################
# import the required module
###############################################################################
from __future__ import division
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from itertools import chain
import pickle
from scipy import linalg
from numpy import dot



###############################################################################
# load  coupon data
###############################################################################
def load_coupon_data():
    coupon_list_train = pd.read_csv("../data/coupon_list_train.csv")
    return coupon_list_train

###############################################################################
# create coupon categorical variable
###############################################################################
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

###############################################################################
# get the list of coupon groups and name for the group
###############################################################################

def get_coupon_id_cluster(X):
#    X.sort_values(by = 'COUPON_ID_hash')
    new_val = X.COUPON_ID_hash.values[0]
    X["coupon_cat"] = new_val
    columns_to_keep = ['COUPON_ID_hash', 'coupon_cat']    
    return X[columns_to_keep]


###############################################################################
# convert coupon_id into cluster
###############################################################################
def get_coupon_id_to_cluster_dict(coupon_id_cluster):
    coupon_id_to_clust_dict = {}
    n_row, n_col = coupon_id_cluster.shape
    for i in xrange(n_row):
        key = coupon_id_cluster.iat[i,0]
        value = coupon_id_cluster.iat[i,1]
        coupon_id_to_clust_dict[key] = value
    return coupon_id_to_clust_dict

###############################################################################
# store coupon_cluster_id definiton in a dictionary
###############################################################################
def get_coupon_clust_def_dict():
    coupon_list_train = load_coupon_data()
    create_coupon_categorical_variable(coupon_list_train)
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
# store the cluster information of all the coupons in a dictionary
###############################################################################
def get_coupon_id_to_cluster_id_dict():
    coupon_list_train = load_coupon_data()
    create_coupon_categorical_variable(coupon_list_train)
    coupon_list_train = coupon_list_train.sort_values(by = 'COUPON_ID_hash')
    coupon_id_cluster = coupon_list_train.groupby(by = ['GENRE_NAME','price_cat', 'price_rate_cat']).apply(get_coupon_id_cluster)
    coupon_id_to_clust_id_dict = get_coupon_id_to_cluster_dict(coupon_id_cluster)
    return coupon_id_to_clust_id_dict

###############################################################################
# get the dictionary that maps coupon id to coupon cluster and coupon cluster to 
# its features
###############################################################################
def get_cluster_info():
    if os.path.isfile('coupon_clust_def_dict.pkl'):
        coupon_clust_def_dict = pickle.load(open('coupon_clust_def_dict.pkl','rb'))
    else:
        coupon_clust_def_dict = get_coupon_clust_def_dict()
        pickle.dump(coupon_clust_def_dict, open('coupon_clust_def_dict.pkl', 'wb '))
    if os.path.isfile('coupon_id_to_clust_id_dict.pkl'):
        coupon_id_to_clust_id_dict = pickle.load(open('coupon_id_to_clust_id_dict.pkl','rb'))
    else:
        coupon_id_to_clust_id_dict = get_coupon_id_to_cluster_id_dict()
        pickle.dump(coupon_id_to_clust_id_dict, open('coupon_id_to_clust_id_dict.pkl', 'wb '))
    return coupon_id_to_clust_id_dict, coupon_clust_def_dict
                

        
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
 #   coupon_visit_selected_users = coupon_visit_selected_users.drop_duplicates(subset = 
 #   ['USER_ID_hash', 'VIEW_COUPON_ID_hash'], keep = 'first') 
    columns_to_keep = ['USER_ID_hash', 'VIEW_COUPON_ID_hash', 'PURCHASE_FLG']
    return coupon_visit_selected_users[columns_to_keep]

# used for creating rating matrix for Sukumar
def create_train_test_set1(n_users = 100, seed_value = 10):
    users_with_purchase = get_users_with_at_least_one_purchase(n_users, seed_value)
    coupon_visit_selected_users = get_visit_data_for_users_with_purchase(users_with_purchase)   
    coupon_visit_selected_users = substitute_coupon_id_with_cluster_id(coupon_visit_selected_users)    
#    n_obs = len(coupon_visit_selected_users)
#    np.random.seed(100)
##    np.random.seed(seed_value)
#    ind_train = np.random.choice(n_obs, size = int(0.7*n_obs), replace = False)
#    ind_test = [x for x in range(n_obs) if x not in ind_train]
#    train = coupon_visit_selected_users.iloc[ind_train]
#    test = coupon_visit_selected_users.iloc[ind_test]
    return coupon_visit_selected_users 

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
# get items purchase during training and testing
###############################################################################    
def item_purchased(train, test):
    purchase_dict = {}
    purchase_dict["train"] = defaultdict(lambda: [])    
    purchase_dict["test"] = defaultdict(lambda: [])
    ind_train = train.PURCHASE_FLG == 1
    ind_test = test.PURCHASE_FLG == 1
    train_purchase = train.loc[ind_train]
    test_purchase = test.loc[ind_test]
    n_train, _ = train_purchase.shape
    n_test, _ = test_purchase.shape
    for i in range(n_train):
        user = train_purchase.USER_ID_hash.iloc[i]
        coupon = train_purchase.VIEW_COUPON_ID_hash.iloc[i]
        purchase_dict["train"][user].append(coupon)
    for i in range(n_test):
        user = test_purchase.USER_ID_hash.iloc[i]
        coupon = test_purchase.VIEW_COUPON_ID_hash.iloc[i]
        purchase_dict["test"][user].append(coupon)
    return purchase_dict


def item_viewed_purchased(train, test):
    view_purchase_dict = {}
    view_purchase_dict['train'] = {}
    view_purchase_dict['test'] = {}
    train = train.sort_values(by = 'PURCHASE_FLG', ascending = False)
    train = train.drop_duplicates(subset = ['USER_ID_hash','VIEW_COUPON_ID_hash','PURCHASE_FLG'], keep = 'first')
    test = test.sort_values(by = 'PURCHASE_FLG', ascending = False)
    test = test.drop_duplicates(subset = ['USER_ID_hash', 'VIEW_COUPON_ID_hash', 'PURCHASE_FLG'], keep = 'first')
    train_users = train.USER_ID_hash.unique().tolist()
    test_users = test.USER_ID_hash.unique().tolist()
    
    n_train, _ = train.shape
    n_test, _ = test.shape
    
    users = list(set(train_users + test_users))
    for user in users:
        view_purchase_dict['train'][user]= {}
        view_purchase_dict['test'][user] = {}
        view_purchase_dict['train'][user]['purchased'] = []
        view_purchase_dict['train'][user]['viewed'] = []
        view_purchase_dict['test'][user]['purchased'] = []
        view_purchase_dict['test'][user]['viewed'] = []

    for i in range(n_train):
        user_id = train.USER_ID_hash.iat[i]
        coupon_id = train.VIEW_COUPON_ID_hash.iat[i]
        purchase_flg = train.PURCHASE_FLG.iat[i]
        if purchase_flg == 1:
            view_purchase_dict['train'][user_id]['purchased'].append(coupon_id)
        if purchase_flg == 0:
            view_purchase_dict['train'][user_id]['viewed'].append(coupon_id)
            
    for i in range(n_test):
        user_id = test.USER_ID_hash.iat[i]
        coupon_id = test.VIEW_COUPON_ID_hash.iat[i]
        purchase_flg = test.PURCHASE_FLG.iat[i]
        if purchase_flg == 1:
            view_purchase_dict['test'][user_id]['purchased'].append(coupon_id)
        if purchase_flg == 0:
            view_purchase_dict['test'][user_id]['viewed'].append(coupon_id)
    
    return view_purchase_dict
    
    
###############################################################################
# create rating matrix. Maybe create later. Ensure that you normalize the rating
# matrix values
###############################################################################
def create_rating_matrix1(train):
    train1 = train.copy()
    train1['rating'] = 0.00
    print "rating column created"
    n_purchase = np.sum(train1.PURCHASE_FLG == 1)
    n_view = np.sum(train1.PURCHASE_FLG == 0)
    view_rating = n_purchase/n_view
    ind = train1.PURCHASE_FLG == 0
    train1.rating.loc[ind] = view_rating
    ind = train1.PURCHASE_FLG == 1
    train1.rating.loc[ind] = 1                     
    train1 = train1.groupby(by = ['USER_ID_hash', 'VIEW_COUPON_ID_hash'], as_index = False ).sum()
    rating_matrix = train1.pivot(index = 'USER_ID_hash', columns = 'VIEW_COUPON_ID_hash', values = 'rating')
    rating_matrix = rating_matrix.fillna(value = 0.0)
    return rating_matrix


def create_rating_matrix2(train):
    train1 = train.copy()
    train1['rating'] = 0.00
    print "rating column created"
    n_purchase = np.sum(train1.PURCHASE_FLG == 1)
    n_view = np.sum(train1.PURCHASE_FLG == 0)
    view_rating = n_purchase/n_view
    ind = train1.PURCHASE_FLG == 0
    train1.loc[ind, "rating"] = view_rating
    ind = train1.PURCHASE_FLG == 1
    train1.loc[ind, "rating"] = 1
    train2 = train1.groupby(by = ['USER_ID_hash', 'VIEW_COUPON_ID_hash', 'PURCHASE_FLG'], as_index = False ).sum()
    ind = train2.rating > 10
    train2.loc[ind, 'rating'] = 10
    train3 = train2.groupby(by = ['USER_ID_hash', 'VIEW_COUPON_ID_hash'], as_index = False ).sum()
    rating_matrix = train3.pivot(index = 'USER_ID_hash', columns = 'VIEW_COUPON_ID_hash', values = 'rating')
    rating_matrix = rating_matrix.fillna(value = 0.0)
    return rating_matrix


def create_rating_matrix3(train):
    train1 = train.copy()
    train1['rating'] = 0.00
    print "rating column created"
    n_purchase = np.sum(train1.PURCHASE_FLG == 1)
    n_view = np.sum(train1.PURCHASE_FLG == 0)
    view_rating = n_purchase/n_view
    ind = train1.PURCHASE_FLG == 0
    train1.rating.loc[ind] = view_rating
    ind = train1.PURCHASE_FLG == 1
    train1.rating.loc[ind] = 1                     
    train1 = train1.groupby(by = ['USER_ID_hash', 'VIEW_COUPON_ID_hash'], as_index = False ).sum()
    rating_matrix = train1.pivot(index = 'USER_ID_hash', columns = 'VIEW_COUPON_ID_hash', values = 'rating')
    rating_matrix = rating_matrix.fillna(value = 0.0)
    return rating_matrix
 
    




def nmf(X, latent_features, max_iter=10000, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    """
    eps = 1e-5
    print 'Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
#    X = X.toarray()  # I am passing in a scipy sparse matrix

    # mask
    mask = np.sign(X)

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(max_iter):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)


        # ==== evaluation ====
        if i % 5 == 0 or i == 1 or i == max_iter:
            print 'Iteration {}:'.format(i),
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')           
            if (curRes < error_limit) or (fit_residual < fit_error_limit):
                break
    return A, Y


###############################################################################
# get recommendation
###############################################################################
def create_final_rating_matrix(rating_matrix, n_comp = 5):
   R = rating_matrix.values
   model = NMF(n_components= n_comp, init='random', random_state=0)
   W = model.fit_transform(R)
   H = model.components_
   R_full = np.dot(W,H)
   return R_full

def create_final_rating_matrix_remove_missing(rating_matrix, n_comp = 5):
   R = rating_matrix.values
   W, H = nmf(R, latent_features = n_comp)
   R_full = np.dot(W,H)
   return R_full


def create_final_rating_matrix_ikeda(rating_matrix, n_comp = 5):
   R = rating_matrix.values
   model = nmf(n_components= n_comp, init='random', random_state=0)
   W = model.fit_transform(R)
   H = model.components_
   R_full = np.dot(W,H)
   return R_full, W, H

def get_rating_matrix_ind_for_test_user(test_user, rating_matrix):
    user_ids = rating_matrix.index.tolist()
    if test_user in user_ids:        
        return user_ids.index(test_user)
    return -1

def get_recommendation_for_user(R_final, rating_matrix, user_ind):
    coupon_clust = rating_matrix.columns
    if user_ind == -1:
        return np.random.choice(coupon_clust, 10)
    user_ratings = R_final[user_ind,]
    sorted_ind = np.argsort(user_ratings,)    
    coupon_clust = list(reversed(coupon_clust[sorted_ind]))
    return coupon_clust[:10]
  
def create_recommendation_for_test_users(test, rating_matrix, R_final):
    user_list = test.USER_ID_hash.tolist()
    test_users_recommendation = defaultdict(lambda: [])
    for i in range(len(user_list)):
        ind = get_rating_matrix_ind_for_test_user(user_list[i], rating_matrix)
        recommendation = get_recommendation_for_user(R_final, rating_matrix, ind)
        test_users_recommendation[user_list[i]] = recommendation
    return test_users_recommendation
    
def create_recommendation_based_on_popular_item(train, test):
    user_list = test.USER_ID_hash.tolist()
    test_users_recommendation = defaultdict(lambda: [])
    top_items = train.groupby(by = 'VIEW_COUPON_ID_hash', as_index = False).sum()
    top_items = top_items.sort_values(by = 'PURCHASE_FLG', ascending = False)['VIEW_COUPON_ID_hash'][:10]
    recommendation = top_items.tolist()
    for i in range(len(user_list)):
        test_users_recommendation[user_list[i]] = recommendation
    return test_users_recommendation
    
    

###############################################################################
# find out percentage accuracy
###############################################################################
def calculate_percentage_accuracy(recommendations_dict, purchase_dict):
    total_bought = 0
    total_correct_recommendation = 0
    for key in recommendations_dict:
        recommendation = recommendations_dict[key]
        purchase = purchase_dict['test'][key]
        total_bought += len(purchase)
        correct_recommendation = [x for x in recommendation if x in purchase]
        total_correct_recommendation += len(correct_recommendation)
    percentage_accuracy = total_correct_recommendation*100/total_bought
    return percentage_accuracy
    
        
        

if __name__ == "__main__":
###############################################################################
# create  dictionaries containing the mapping of coupon id to coupon cluster and
# coupon cluster to coupon cluster description
###############################################################################
    coupon_id_to_clust_id_dict = get_coupon_id_to_cluster_id_dict()
    coupon_clust_def_dict = get_coupon_clust_def_dict() 
###############################################################################
# saving the results in the pickle files
###############################################################################    
    pickle.dump(coupon_id_to_clust_id_dict, open('coupon_id_to_clust_id_dict.pkl', 'wb '))
    pickle.dump(coupon_clust_def_dict, open('coupon_clust_def_dict.pkl', 'wb '))
###############################################################################
# reading the data from the pickled object
###############################################################################    
    coupon_id_to_clust_id_dict1 = pickle.load(open('coupon_id_to_clust_id_dict.pkl','rb'))
    coupon_clust_def_dict1 = pickle.load(open('coupon_clust_def_dict.pkl','rb'))
    
    train, test = create_train_test_set(n_users = 100, seed_value = 10)

# testing the module    
    for key, value in coupon_id_to_clust_id_dict.items():
        print coupon_clust_def_dict[value]
    for key in coupon_clust_def_dict:
        print key, coupon_clust_def_dict[key][0], coupon_clust_def_dict[key][1], coupon_clust_def_dict[key][2] 
        
    purchase_dict = item_purchased(train, test) 
  