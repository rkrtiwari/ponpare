# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 10:47:35 2017

@author: tiwarir
"""
###############################################################################
# import the required module
###############################################################################
from __future__ import division
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import create_training_test_data as cttd
import subset_data as sd
import data_preprocessing as dpre
import random
import matplotlib.pyplot as plt

###############################################################################
# setting display options
###############################################################################
pd.options.display.max_columns = 100
pd.set_option('expand_frame_repr', False)
    
###############################################################################
# create rating matrix. Maybe create later. Ensure that you normalize the rating
# matrix values
###############################################################################

def create_rating_matrix(train):
    train1 = train.copy()
    train1['rating'] = 0.00
    n_purchase = np.sum(train1.PURCHASE_FLG == 1)
    n_view = np.sum(train1.PURCHASE_FLG == 0)
    view_rating = n_purchase/n_view
    ind = train1.PURCHASE_FLG == 0
    train1.rating.loc[ind] = view_rating
    ind = train1.PURCHASE_FLG == 1
    train1.rating.loc[ind] = 1                     
    train2 = train1.groupby(by = ['USER_ID_hash', 'VIEW_COUPON_ID_hash'], as_index = False ).sum()
    rating_matrix = train2.pivot(index = 'USER_ID_hash', columns = 'VIEW_COUPON_ID_hash', values = 'rating')
    rating_matrix = rating_matrix.fillna(value = 0.0)
    return rating_matrix

def create_final_rating_matrix(rating_matrix, n_comp = 5):
   R = rating_matrix.values
   model = NMF(n_components= n_comp, init='random', random_state=0)
   W = model.fit_transform(R)
   H = model.components_
   R_full = np.dot(W,H)
   final_rating_matrix = pd.DataFrame(R_full, index = rating_matrix.index,
                                   columns = rating_matrix.columns)
   return final_rating_matrix


def get_recommendation_for_a_user(final_rating_matrix, user_id):
    coupon_clust = final_rating_matrix.columns.tolist()
    training_users = final_rating_matrix.index.tolist() 
    
    if user_id not in training_users:
        return np.random.choice(coupon_clust, 10)
    
    user_ratings = final_rating_matrix.loc[user_id]
    user_ratings = user_ratings.sort_values(ascending = False)
    return user_ratings.index[:10].tolist()

def get_recommendation_for_test_data(test, final_rating_matrix):
    test_users = test.USER_ID_hash.unique().tolist()
    test_users_recommendation = {}
    
    for user in test_users:
        recommendation = get_recommendation_for_a_user(final_rating_matrix, user)
        test_users_recommendation[user] = recommendation
                                 
    return test_users_recommendation
        

def get_collaborative_filtering_recommendation(train, test):
    rating_matrix = create_rating_matrix(train)
    final_rating_matrix = create_final_rating_matrix(rating_matrix, n_comp = 5)
    test_users_recommendation = get_recommendation_for_test_data(test, final_rating_matrix)
    return test_users_recommendation
    
if __name__ == '__main__':
    seed_value = random.choice(range(9999))
    coupon_visit_subset = sd.create_data_subset(n_users = 5, min_purchase = 1, max_purchase = 10, seed_value = seed_value)
    coupon_clust_visit = dpre.substitute_coupon_id_with_cluster_id(coupon_visit_subset)
    train, test = cttd.create_train_test_set(coupon_clust_visit, train_frac = 0.7, seed_value = 10)
    rating_matrix = create_rating_matrix(train)
    final_rating_matrix = create_final_rating_matrix(rating_matrix, n_comp = 5)
    rating_matrix_val = rating_matrix.values
    final_rating_matrix_val = final_rating_matrix.values
    print "Initial Rating Matrix"
    plt.matshow(rating_matrix_val)
    plt.show()
    print "Final Rating Matrix"
    plt.matshow(final_rating_matrix_val)
    plt.show()
    test_users_recommendation = get_recommendation_for_test_data(test, final_rating_matrix)
    test_users_recommendation_using_main_function = get_collaborative_filtering_recommendation(train, test)
    test_users = test.USER_ID_hash.unique().tolist()
    for user in test_users:
        if user not in final_rating_matrix.index:
            continue           
        ratings = final_rating_matrix.loc[user].sort_values(ascending = False)
        plt.plot(ratings.values[:10])
        plt.show()
        rec1 = ratings[:10]
        rec2 = test_users_recommendation[user]
        rec3 = test_users_recommendation_using_main_function[user]
        rec_dict = {"rec2": pd.Series(rec2), "rec3": pd.Series(rec3), "rec1": pd.Series(rec1.index), "rec1_values": pd.Series(rec1.values)}
        df = pd.DataFrame(rec_dict)
        print df
        if rec1.index.tolist() == rec2:
            print "____________________________________________________________"
            print "REC1 and REC2 ARE THE SAME "
            print "____________________________________________________________"
        else:
            print "PROBLEM WITH THE RECOMMENDATION. See REC1 and REC2"
            
        if rec2 == rec3:
            print "____________________________________________________________"
            print "REC2 and REC3 ARE THE SAME"
            print "____________________________________________________________"
        else:
            print "PROBLEM WITH THE RECOMMENDATION. See REC2 and REC3"
            
        
        
        
        
    
    
    
