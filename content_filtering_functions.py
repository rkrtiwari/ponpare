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
import pickle


###############################################################################
# Functions
###############################################################################
def load_input_files():
    user_list = pd.read_csv("data/user_list.csv")
    coupon_list_train = pd.read_csv("data/coupon_list_train.csv")
    return user_list, coupon_list_train

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


def create_user_vector_dict(user_list):
#    user_content_vector_dict = defaultdict(dd)
    user_content_vector_dict = {}
    n_users, n_features = user_list.shape
    
    for i in range(n_users):
        user_id = user_list.USER_ID_hash.iloc[i]
        pref = user_list.PREF_NAME.iloc[i]
        gender = user_list.SEX_ID.iloc[i]
        age = user_list.age_cat.iloc[i]
        user_vector = user_vector_from_user_content(age, gender)
        user_content_vector_dict[user_id] = [user_vector, pref]
    pickle.dump(user_content_vector_dict, open('user_content_vector_dict.pkl', 'wb '))
    return user_content_vector_dict

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

def create_coupon_vector_dict(coupon_list_train, coupon_cond_prob):
    coupon_content_vector_dict = {}
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
        c_u_vector = coupon_feature_to_user_vector(c_features, user_features, coupon_cond_prob)
        coupon_content_vector_dict[c_id] = [c_u_vector, area, validity]
    pickle.dump(coupon_content_vector_dict, open('coupon_content_vector_dict.pkl', 'wb '))
    return coupon_content_vector_dict


def get_purchased_coupon_data():
    coupon_visit_train = pd.read_csv("data/coupon_visit_train.csv")
    pur_ind = coupon_visit_train.PURCHASE_FLG == 1
    purchased_coupons = coupon_visit_train[pur_ind]
    columns_to_keep = ['I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash']
    purchased_coupons = purchased_coupons[columns_to_keep]
    return purchased_coupons

def merge_purchased_coupon_to_get_location(purchased_coupons):
    user_list = pd.read_csv("data/user_list.csv")
    coupon_list_train = pd.read_csv("data/coupon_list_train.csv")
    
    purchased_user_info = purchased_coupons.merge(user_list, how = 'left', 
                                                     on = 'USER_ID_hash')
    purchased_user_coupon_info = purchased_user_info.merge(coupon_list_train,
                how = 'left', left_on = 'VIEW_COUPON_ID_hash', right_on = 'COUPON_ID_hash')
    columns_to_keep = ['I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash', 'PREF_NAME',
                   'ken_name']
    
    purchased_user_coupon = purchased_user_coupon_info[columns_to_keep]
    return purchased_user_coupon

def get_purchased_coupon_area_by_user_area(purchased_user_coupon):
    purchased_coupon_ken_by_user_pref = pd.DataFrame(columns = ['pref', 'ken',
                                                         'count', 'per_purchase'])
    purchased_user_coupon.dropna(axis=0, how = 'any', inplace = True)
    user_prefs = purchased_user_coupon.PREF_NAME.unique()
    for pref in user_prefs:
        ind = purchased_user_coupon.PREF_NAME == pref
        df = purchased_user_coupon.loc[ind]
        df_value_counts = df.ken_name.value_counts()
        n_entry = len(df_value_counts)
        pref_s = pd.Series(index=range(n_entry))        
        for i in range(n_entry):
            pref_s.loc[i] = pref
        ken = pd.Series(df_value_counts.index)
        count = pd.Series(df_value_counts.values)
        per = count*100/np.sum(count)
        df_2 = pd.DataFrame({'pref':pref_s, 'ken': ken, 'count': count, 'per_purchase': per })
        purchased_coupon_ken_by_user_pref = pd.concat([purchased_coupon_ken_by_user_pref, df_2], ignore_index=True)
        
    purchased_coupon_ken_by_user_pref.sort_values(by = ['pref', 'count'], 
                                              inplace = True, ascending = False)  
    return purchased_coupon_ken_by_user_pref

def get_top_coupon_area_for_user_area(purchased_coupon_ken_by_user_pref, n = 3):
    purchase_area = {}
    pref_names = purchased_coupon_ken_by_user_pref.pref.unique()
    for pref in pref_names:
        ind = purchased_coupon_ken_by_user_pref.pref == pref
        df = purchased_coupon_ken_by_user_pref.loc[ind]
        if df.empty:
            continue
        ken_info = {}
        for i in range(n):
            key = df.ken.values[i]
            value = df.per_purchase.values[i]
            ken_info[key] = value
            
        purchase_area[pref] = ken_info        
    return purchase_area

def get_user_purchase_area(n = 3):
    purchased_coupons =  get_purchased_coupon_data() 
    purchased_user_coupon = merge_purchased_coupon_to_get_location(purchased_coupons)
    purchased_coupon_ken_by_user_pref = get_purchased_coupon_area_by_user_area(purchased_user_coupon)
    purchase_area = get_top_coupon_area_for_user_area(purchased_coupon_ken_by_user_pref, n = n)
    pickle.dump(purchase_area, open('purchase_area_dict.pkl', 'wb '))
    return purchase_area
    
    
#def check_result(purchased_user_coupon, purchase_area):
#    pref_names = purchased_user_coupon.PREF_NAME.unique()
#    pref_names = purchased_user_coupon.PREF_NAME.unique()
#    n_pref = len(pref_names)
#    ind = np.random.choice(n_pref)
#    pref = pref_names[ind]
#    pref_ind = purchased_user_coupon.PREF_NAME == pref
#    df = purchased_user_coupon.loc[pref_ind]
#    df = df[['ken_name', 'PREF_NAME']].groupby('ken_name').agg('count')
#    df = df.sort_values(by = 'PREF_NAME', ascending = False)
#    df['percentage'] = df['PREF_NAME']*100/np.sum(df['PREF_NAME'])
#    print df[:3]
#    
#    for i in range(3):
#        for key in purchase_area[pref][i]:
#            print key, purchase_area[pref][i][key]
#        
#
#check_result(purchased_user_coupon, purchase_area)


#for key in purchase_area.keys():
#    print key, purchase_area[key]
    





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
def get_coupon_clust_def_dict(coupon_list_train):
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

def get_recommendation(user_info, coupon_content_vector_dict, purchased_area):
    user_id = user_info[0]
    user_content = user_info[1]
    product_rank = create_product_ranking(user_content, coupon_content_vector_dict, purchased_area)
    product_rank = product_rank.drop_duplicates(subset = 'score', keep = 'first')
    coupons = product_rank.coupon_id.iloc[:10]
#    recommendation = []
#    for coupon in coupons:
#        ind = coupon_list_train.COUPON_ID_hash == coupon
#        coupon_feat = (coupon_list_train.price_rate_cat.loc[ind].values[0], \
#          coupon_list_train.price_cat.loc[ind].values[0],\
#          coupon_list_train.GENRE_NAME.loc[ind].values[0])
#        recommendation.append(coupon_feat)
    return coupons

def get_items_bought(user_id):
    purchased = []
    ind = test.USER_ID_hash == user_id
    actual_purchase = test[['GENRE_NAME', 'price_cat', 'price_rate_cat']].loc[ind]
    for i in range(len(actual_purchase)):
        pur_coup_feat = (actual_purchase.price_rate_cat.values[i], \
                     actual_purchase.price_cat.values[i],\
                     actual_purchase.GENRE_NAME.values[i])
        purchased.append(pur_coup_feat)
    return purchased

def get_accuracy_multiple_users(n=5):
    accuracy = []
    for i in xrange(n):
        print i
        user_id, user_content = get_a_user(user_list, user_content_vector_dict, test)
        recommendation = get_recommendation(user_id, user_content)
        purchased =  get_items_bought(user_id)
        accuracy.append(check_accuracy(recommendation, purchased))
    return accuracy



if __name__ == "__main__":
    user_vector_from_user_content( 'age1','f')    




