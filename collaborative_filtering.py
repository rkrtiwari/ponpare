# -*- coding: utf-8 -*-
"""
Created on Tue May 23 08:26:49 2017

@author: tiwarir
"""
###############################################################################
# import the required module
###############################################################################
import os
import numpy as np
import pandas as pd
from __future__ import division

###############################################################################
# setting display options
###############################################################################
pd.options.display.max_columns = 100
pd.set_option('expand_frame_repr', False)

###############################################################################
# change to the appropriate working directory
###############################################################################

os.getcwd()
new_dir = os.path.join("Documents", "ponpare")
os.chdir(new_dir)

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

coupon_list_train = pd.read_csv("data/coupon_list_train.csv") 
create_coupon_categorical_variable(coupon_list_train)   
coupon_list_train.head()

################################################################################
## finding theoretical number of coupon groups and actual number of coupon groups
################################################################################
#def find_expected_and_actual_coupon_types(coupon_list_train):
#    n_genre = len(coupon_list_train.GENRE_NAME.unique())
#    n_price_cat = len(coupon_list_train.price_cat.unique())
#    n_price_rate_cat = len(coupon_list_train.price_rate_cat.unique())
#    n_expected = n_genre*n_price_cat*n_price_rate_cat
#    coupon_group_list = coupon_list_train.sort_values(by = ['COUPON_ID_hash',
#    'GENRE_NAME','price_cat', 'price_rate_cat'])
#    coupon_group_list = coupon_group_list.drop_duplicates(subset = ['GENRE_NAME','price_cat', 
#    'price_rate_cat'], keep =  'first')
#    n_actual = len(coupon_group_list)
#    return n_expected, n_actual
#
#find_expected_and_actual_coupon_types(coupon_list_train)

###############################################################################
# get the list of coupon groups and name for the group
# change the name. this function name is not very reflective of what it is
# doing
###############################################################################
def get_coupon_groups(coupon_list_train):    
    coupon_group_list = coupon_list_train.sort_values(by = ['COUPON_ID_hash',
    'GENRE_NAME','price_cat', 'price_rate_cat'])
    coupon_group_list = coupon_group_list.drop_duplicates(subset = ['GENRE_NAME','price_cat', 
    'price_rate_cat'], keep =  'first')
    columns_to_keep = ['COUPON_ID_hash','GENRE_NAME', 'price_cat', 'price_rate_cat']
    coupon_group_list = coupon_group_list[columns_to_keep]    
    return coupon_group_list

coupon_group_list = get_coupon_groups(coupon_list_train)

###############################################################################
# get the list of coupon groups and name for the group
###############################################################################

def get_coupon_id_cluster(X):
    X.sort_values(by = 'COUPON_ID_hash')
    new_val = X.COUPON_ID_hash.values[0]
    X["coupon_cat"] = new_val
    columns_to_keep = ['COUPON_ID_hash', 'coupon_cat']    
    return X[columns_to_keep]

coupon_id_cluster = coupon_list_train.groupby(by = ['GENRE_NAME','price_cat', 'price_rate_cat']).apply(get_coupon_id_cluster)

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
                             
        
coupon_id_to_clust_dict = get_coupon_id_to_cluster_dict(coupon_id_cluster)

##############################################################################
# testing coupon_id to coupon_cluster conversion
###############################################################################

# module to check the result
# prints the feature of the randomly chosen coupon features with the feature 
# of the cluster that it has been assigned to it
#def check_coupon_cluster_assignment(coupon_id_cluster, coupon_list_train):
#    n = len(coupon_id_cluster)
#    i = np.random.choice(range(n))
#    coupon_id = coupon_id_cluster.iat[i,0]
#    coupon_cat = coupon_id_cluster.iat[i,1]
#    ind1 = coupon_list_train.COUPON_ID_hash == coupon_id
#    ind2 = coupon_list_train.COUPON_ID_hash == coupon_cat
#    print coupon_list_train['COUPON_ID_hash'].loc[ind1]
#    print coupon_list_train['COUPON_ID_hash'].loc[ind2]
#    print '\n'
#                                    
#    print coupon_list_train['price_cat'].loc[ind1]
#    print coupon_list_train['price_cat'].loc[ind2] 
#    print '\n'
#    
#    print coupon_list_train['price_rate_cat'].loc[ind1]
#    print coupon_list_train['price_rate_cat'].loc[ind2]
#    print '\n'
#    
#    print coupon_list_train['GENRE_NAME'].loc[ind1]
#    print coupon_list_train['GENRE_NAME'].loc[ind2]
#        
#check_coupon_cluster_assignment(coupon_id_cluster, coupon_list_train)    

# checking coupon cluster assignement dictionary version

def testing_coupon_cluster_assignement(coupon_id_to_clust_dict, coupon_list_train):
    key = np.random.choice(coupon_id_to_clust_dict.keys())
    value = coupon_id_to_clust_dict[key]
    ind1 = coupon_list_train.COUPON_ID_hash == key
    ind2 = coupon_list_train.COUPON_ID_hash == value
    columns_to_keep = ['COUPON_ID_hash', 'GENRE_NAME','price_rate_cat', 'price_cat']
    print "coupon_id"
    print coupon_list_train[columns_to_keep].loc[ind1]
    print "coupon_cluster_id"
    print coupon_list_train[columns_to_keep].loc[ind2]
    
    
testing_coupon_cluster_assignement(coupon_id_to_clust_dict, coupon_list_train)

###############################################################################
# replacing coupon_id with cluster_id
###############################################################################
def replace_coupon_id_with_cluster_id(coupon_id_to_clust_dict):
    coupon_visit_train = pd.read_csv("data/coupon_visit_train.csv") 
    columns_to_keep = ['I_DATE', 'PURCHASE_FLG', 'VIEW_COUPON_ID_hash', 'USER_ID_hash']
    coupon_clust_visit = coupon_visit_train[columns_to_keep].copy()

    
    for i in range(len(coupon_clust_visit)):
        coupon_id = coupon_clust_visit.iat[i,1]
        if coupon_id in coupon_id_to_clust_dict:
            coupon_cat = coupon_id_to_clust_dict[coupon_id]
        else:
            coupon_cat = -1
        coupon_clust_visit.ix[i, 'VIEW_COUPON_ID_hash'] = coupon_cat
    ind = coupon_clust_visit.VIEW_COUPON_ID_hash == -1
    coupon_clust_visit = coupon_clust_visit.loc[~ind, ]
    coupon_clust_visit = coupon_clust_visit.sort_values(by = 'PURCHASE_FLG', ascending = False)
    coupon_clust_visit = coupon_clust_visit.drop_duplicates(subset = ['VIEW_COUPON_ID_hash', 'USER_ID_hash'],
                                       keep = 'first')
    return coupon_clust_visit

    
coupon_clust_visit = replace_coupon_id_with_cluster_id(coupon_id_to_clust_dict) 



def 

coupon_visit_train = pd.read_csv("data/coupon_visit_train.csv")
columns_to_keep = ['I_DATE', 'PURCHASE_FLG', 'VIEW_COUPON_ID_hash', 'USER_ID_hash']
coupon_clust_visit = coupon_visit_train[columns_to_keep].copy()
coupon_clust_visit = coupon_clust_visit.sort_values(by = 'PURCHASE_FLG', ascending = False)
coupon_clust_visit = coupon_clust_visit.drop_duplicates(subset = ['PURCHASE_FLG', 
                     'VIEW_COUPON_ID_hash', 'USER_ID_hash'], keep = 'first')
pur_ind = coupon_clust_visit.PURCHASE_FLG == 1    
user_with_purchase = coupon_clust_visit.loc[pur_ind].USER_ID_hash.unique()
len(user_with_purchase)
ind_user_purchase = coupon_clust_visit.USER_ID_hash.isin(user_with_purchase)
coupon_clust_visit = coupon_clust_visit[ind_user_purchase]
coupon_list_train = pd.read_csv("data/coupon_list_train.csv")
listed_coupons = coupon_list_train.COUPON_ID_hash.unique()
ind_listed_coupon = coupon_clust_visit.VIEW_COUPON_ID_hash.isin(listed_coupons)
coupon_clust_visit = coupon_clust_visit[ind_listed_coupon]
coupon_clust_visit.columns









###############################################################################
# Find out actually what percentage of people buy. Make subsetting so that we have
# data for only those people. Other way of subseting is not working
###############################################################################
def get_users_with_at_least_one_purchase(coupon_clust_visit, n=50):
    ind_pur = coupon_clust_visit.PURCHASE_FLG == 1
    user_ids = coupon_clust_visit.loc[ind_pur].USER_ID_hash.unique()
    n_users = len(user_ids)
    ind = np.random.choice(range(n_users), size = n)
    return user_ids[ind]
    
users_with_purchase = get_users_with_at_least_one_purchase(coupon_clust_visit, n=50)

# checking the code to make sure that a random user has indeed made at least one
# purchase
#def check_users_with_purchase(users_with_purchase):
#    n_users = len(users_with_purchase)
#    ind = np.random.choice(range(n_users))
#    user_id = users_with_purchase[ind]
#    coupon_visit_train = pd.read_csv("data/coupon_visit_train.csv")
#    user_ind = coupon_visit_train.USER_ID_hash == user_id
#    coupon_visit_train = coupon_visit_train.loc[user_ind]
#    columns_to_keep = ['PURCHASE_FLG', 'USER_ID_hash', 'VIEW_COUPON_ID_hash']
#    coupon_visit_train = coupon_visit_train[columns_to_keep]    
#    coupon_visit_train = coupon_visit_train.sort_values(by = 'PURCHASE_FLG', ascending = False)
#    return coupon_visit_train.iloc[:5]
#
#check_users_with_purchase(users_with_purchase)

###############################################################################
# get viewing/purchasing behavior of those users
###############################################################################
def get_visit_data_for_users_with_purchase(coupon_clust_visit, users_with_purchase):
    ind = coupon_clust_visit.USER_ID_hash.isin(users_with_purchase)
    coupon_clust_visit_select_users = coupon_clust_visit.loc[ind]
    return coupon_clust_visit_select_users
    
coupon_clust_visit_selected_users =  get_visit_data_for_users_with_purchase(coupon_clust_visit, users_with_purchase)

def create_train_test_set(coupon_cluster_visit_selected_users):
    n_obs = len(coupon_cluster_visit_selected_users)
    ind_train = np.random.choice(n_obs, size = int(0.7*n_obs))
    ind_test = [x for x in range(n_obs) if x not in ind_train]
    train = coupon_cluster_visit_selected_users.iloc[ind_train]
    test = coupon_cluster_visit_selected_users.iloc[ind_test]
    return train, test
    
train, test = create_train_test_set(coupon_clust_visit_selected_users)    

def create_rating_matrix(train):
    ind_seen = train.PURCHASE_FLG == 0
    ind_pur = train.PURCHASE_FLG == 1
    train.loc[ind_seen, 'rating'] = 0.7
    train.loc[ind_pur, 'rating'] = 1
    rating_matrix = train.pivot(index = 'USER_ID_hash', columns = 'VIEW_COUPON_ID_hash', values = 'rating')
    return rating_matrix
    
create_rating_matrix(train)

def matrix_factorization_and_rating_matrix_reconstruction(R, K = 2, steps=5000, alpha=0.0002, beta=0.02):
    
    N = len(R)
    M = len(R[0])

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

# testing matrix factorization
R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

R = np.array(R)
R_full = np.dot(nP, nQ.T)
user_id = 2 ## Recommendation required for the 2nd user
    
user_rating = R_full[user_id -1 , ]
ind = R[user_id -1, ] != 0
user_rating[ind] = 0
rec_coupon = np.argmax(user_rating) + 1
print(rec_coupon)    ## Recommended coupon for the user with user_id = 2

# testing if the user indeed bought the recommended item

###############################################################################
# get viewing/purchasing behavior of those users
###############################################################################
#def get_visit_data_for_users_with_purchase(users_with_purchase):
#    coupon_visit_train = pd.read_csv("data/coupon_visit_train.csv")
#    ind = coupon_visit_train.USER_ID_hash.isin(users_with_purchase)
#    coupon_visit_train = coupon_visit_train.loc[ind]
#    coupon_visit_train = coupon_visit_train.sort_values(by = 'PURCHASE_FLG', 
#                                                        ascending = False )
#    coupon_visit_train = coupon_visit_train.drop_duplicates(subset = [ 'VIEW_COUPON_ID_hash', 
#    'USER_ID_hash'], keep = 'first')
#    return coupon_visit_train
#    
#coupon_visit_selected_users =  get_visit_data_for_users_with_purchase(users_with_purchase)


###############################################################################
# substitute selected users coupon id with coupon cluster id
###############################################################################
#def substitute_coupon_id_with_cluster_id(coupon_visit_selected_users, coupon_id_to_clust_dict):
#    coupons_in_dict = coupon_id_to_clust_dict.keys()
#    ind = coupon_visit_selected_users.VIEW_COUPON_ID_hash.isin(coupons_in_dict)
#    coupon_visit_selected_users = coupon_visit_selected_users.loc[ind]
#    n = len(coupon_visit_selected_users)
#    for i in range(n):
#        coupon_id = coupon_visit_selected_users.VIEW_COUPON_ID_hash.iat[i]
#        coupon_visit_selected_users.VIEW_COUPON_ID_hash.iat[i] = coupon_id_to_clust_dict[coupon_id]
#    coupon_visit_selected_users = coupon_visit_selected_users.sort_values(by = 'PURCHASE_FLG', ascending = False)
#    coupon_visit_selected_users = coupon_visit_selected_users.drop_duplicates(subset = 
#    ['USER_ID_hash', 'VIEW_COUPON_ID_hash'], keep = 'first') 
#    columns_to_keep = ['USER_ID_hash', 'VIEW_COUPON_ID_hash', 'PURCHASE_FLG']
#    return coupon_visit_selected_users[columns_to_keep]
        
#coupon_cluster_visit_selected_users = substitute_coupon_id_with_cluster_id(coupon_visit_selected_users, coupon_id_to_clust_dict)  

###############################################################################
# find out the regions people mostly buy from (this way of subseting not working)
###############################################################################
# 1. subsetting data to get only the 
#def get_purchased_coupon_data():
#    coupon_visit_train = pd.read_csv("data/coupon_visit_train.csv")    
#    pur_ind = coupon_visit_train.PURCHASE_FLG == 1
#    purchased_coupons = coupon_visit_train.loc[pur_ind]
#    columns_to_keep = ['I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash']
#    purchased_coupons = purchased_coupons[columns_to_keep]
#    return purchased_coupons
#
#    
#purchased_coupons =  get_purchased_coupon_data()       
#purchased_coupons.head()
#
#def merge_purchased_coupon_to_get_location(purchased_coupons):
#    user_list = pd.read_csv('data/user_list.csv')
#    coupon_list_train = pd.read_csv('data/coupon_list_train.csv')
#    
#    purchased_user_info = purchased_coupons.merge(user_list, how = 'left', 
#                                                     on = 'USER_ID_hash')
#    purchased_user_coupon_info = purchased_user_info.merge(coupon_list_train,
#                how = 'left', left_on = 'VIEW_COUPON_ID_hash', right_on = 'COUPON_ID_hash')
#    columns_to_keep = ['I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash', 'PREF_NAME',
#                   'ken_name']
#    
#    purchased_user_coupon = purchased_user_coupon_info[columns_to_keep]
#    return purchased_user_coupon
#    
#purchased_user_coupon_info = merge_purchased_coupon_to_get_location(purchased_coupons)

###############################################################################
# getting the coupon purchase area count by user prefecture
###############################################################################

#def get_purchased_coupon_area_by_user_area(purchased_user_coupon_info):
#    purchased_coupon_ken_by_user_pref = pd.DataFrame(columns = ['pref', 'ken',
#                                                         'count', 'per_purchase'])
#    purchased_user_coupon_info.dropna(axis=0, how = 'any', inplace = True)
#    user_prefs = purchased_user_coupon_info.PREF_NAME.unique()
#    for pref in user_prefs:
#        ind = purchased_user_coupon_info.PREF_NAME == pref
#        df = purchased_user_coupon_info.loc[ind]
#        df_value_counts = df.ken_name.value_counts()
#        n_entry = len(df_value_counts)
#        pref_s = pd.Series(index=range(n_entry))        
#        for i in range(n_entry):
#            pref_s.loc[i] = pref
#        ken = pd.Series(df_value_counts.index)
#        count = pd.Series(df_value_counts.values)
#        per = count*100/np.sum(count)
#        df_2 = pd.DataFrame({'pref':pref_s, 'ken': ken, 'count': count, 'per_purchase': per })
#        purchased_coupon_ken_by_user_pref = pd.concat([purchased_coupon_ken_by_user_pref, df_2], ignore_index=True)
#        
#    purchased_coupon_ken_by_user_pref.sort_values(by = ['pref', 'count'], 
#                                              inplace = True, ascending = False)  
#    return purchased_coupon_ken_by_user_pref
#
#purchased_coupon_ken_by_user_pref = get_purchased_coupon_area_by_user_area(purchased_user_coupon_info)
###############################################################################
# find out number of users in a given prefecture
###############################################################################

#def get_top_coupon_area_for_user_area(purchased_coupon_ken_by_user_pref, n = 5):
#    purchase_area = {}
#    pref_names = purchased_coupon_ken_by_user_pref.pref.unique()
#    for pref in pref_names:
#        ind = purchased_coupon_ken_by_user_pref.pref == pref
#        df = purchased_coupon_ken_by_user_pref.loc[ind]
#        if df.empty:
#            continue
#        ken_info = {}
#        for i in range(n):
#            key = df.ken.values[i]
#            value = df.per_purchase.values[i]
#            ken_info[key] = value
#            
#        purchase_area[pref] = ken_info
#        
#    return purchase_area
#
#purchased_coupon_area = get_top_coupon_area_for_user_area(purchased_coupon_ken_by_user_pref)

## find out number of users in each prefecture
#def get_user_count_in_pref():
#    user_list = pd.read_csv('data/user_list.csv')
#    user_count_in_pref = user_list.PREF_NAME.value_counts()
#    return user_count_in_pref
#
#user_count_in_pref = get_user_count_in_pref()

## choose a prefecture that will be used in collaborative filtering based on the
## number of users
#def get_a_pref_based_on_number_of_users(user_count_in_pref, n_users = 250):
#    user_count_in_pref.sort_values(ascending = False)
#    ind = user_count_in_pref <= n_users
#    return user_count_in_pref.loc[ind, ].index[0]
#    
#pref = get_a_pref_based_on_number_of_users(user_count_in_pref, n_users = 1000)
#print pref, user_count_in_pref.loc[pref]   
#    
### find the coupon ken that users from this prefecture usually buys from
#def get_coupon_ken_for_user_pref(pref, purchased_coupon_area):
#    return purchased_coupon_area[pref].keys()
#
#coupon_kens = get_coupon_ken_for_user_pref(pref, purchased_coupon_area)
#print pref 
#for ken in coupon_kens:
#    print ken
  
###############################################################################
# do the subsetting to get the visit data for  users of a given prefecture and 
# only from the top 3 kens they usually buy from
###############################################################################
#def subset_coupon_visit_based_on_user_pref_and_coupon_ken(pref, coupon_kens):
#    coupon_visit_train = pd.read_csv("data/coupon_visit_train.csv") 
#    coupon_list_train = pd.read_csv("data/coupon_list_train.csv")
#    user_list = pd.read_csv('data/user_list.csv')
#    coupon_list_train = pd.read_csv('data/coupon_list_train.csv')
#    user_visit_info = coupon_visit_train.merge(user_list, how = 'left', 
#                                                     on = 'USER_ID_hash')
#    user_visit_coupon_info = user_visit_info.merge(coupon_list_train,
#                how = 'left', left_on = 'VIEW_COUPON_ID_hash', right_on = 'COUPON_ID_hash')
#    columns_to_keep = ['I_DATE', 'PURCHASE_FLG','VIEW_COUPON_ID_hash', 'USER_ID_hash', 'PREF_NAME',
#                   'ken_name']
#    
#    visited_coupon_user = user_visit_coupon_info[columns_to_keep]
#    ind1 = visited_coupon_user.PREF_NAME == pref
#    ind2 = coupon_list_train.ken_name.isin(coupon_kens)
#    ind = ind1 & ind2
#    visited_coupon_user = visited_coupon_user.loc[ind]    
#    return visited_coupon_user
#    
    
#coupon_visit_of_a_pref = subset_coupon_visit_based_on_user_pref_and_coupon_ken(pref, coupon_kens)
#coupon_visit_of_a_pref.head(15)
#coupon_visit_of_a_pref.shape
#coupon_visit_of_a_pref.head()
#np.sum(coupon_visit_of_a_pref.PURCHASE_FLG)
#
#
## checking the result of subsetting
#coupon_visit_train = pd.read_csv("data/coupon_visit_train.csv")
#coupon_visit_train.columns
#np.sum(coupon_visit_train.PURCHASE_FLG)*100/len(coupon_visit_train.PURCHASE_FLG)
#len(coupon_visit_train.PURCHASE_FLG)
# 

#############################################################################
# Updating rating matrix when a user views a coupon
#############################################################################

 
                

        
    R_full = np.dot(P, Q)
    uid = get_existing_users_collaborative_filtering()
    cid = get_existing_coupons_collaborative_filtering()
    R_full = pd.DataFrame(R_full, index = uid, columns = cid)
    
    conn = sqlite3.connect("user_info.db")
    R_full.to_sql("reconstructed_rating_matrix", conn, if_exists = "replace")
    conn.close()  
                     











