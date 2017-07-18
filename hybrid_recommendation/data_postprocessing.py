# -*- coding: utf-8 -*-
"""
Created on Mon Jul 03 12:08:30 2017

@author: tiwarir
"""


###############################################################################
# import the required module
###############################################################################
from __future__ import division 
import subset_data as sd
import create_training_test_data as cttd
import data_preprocessing as dpre 
import random  
import collaborative_filtering_functions as colf
import matplotlib.pyplot as plt
import numpy as np
import data_loading as dl
import os
import pandas as pd
###############################################################################
# get items purchase during training and testing
###############################################################################    

def item_viewed_purchased(train, test):
    view_purchase_dict = {}
    view_purchase_dict['train'] = {}
    view_purchase_dict['test'] = {}

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
    
    for i in range(n_train):
        user_id = train.USER_ID_hash.iat[i]
        coupon_id = train.VIEW_COUPON_ID_hash.iat[i]
        purchase_flg = train.PURCHASE_FLG.iat[i]
        if purchase_flg == 0:
            if coupon_id not in view_purchase_dict['train'][user_id]['purchased']:
                view_purchase_dict['train'][user_id]['viewed'].append(coupon_id)
                   
    for i in range(n_test):
        user_id = test.USER_ID_hash.iat[i]
        coupon_id = test.VIEW_COUPON_ID_hash.iat[i]
        purchase_flg = test.PURCHASE_FLG.iat[i]
        if purchase_flg == 1:
            view_purchase_dict['test'][user_id]['purchased'].append(coupon_id)
            
    for i in range(n_test):
        user_id = test.USER_ID_hash.iat[i]
        coupon_id = test.VIEW_COUPON_ID_hash.iat[i]
        purchase_flg = test.PURCHASE_FLG.iat[i]
        if purchase_flg == 0:
            if coupon_id not in view_purchase_dict['test'][user_id]['purchased']:
                view_purchase_dict['test'][user_id]['viewed'].append(coupon_id)
                
    return view_purchase_dict


def calculate_percentage_accuracy(recommendations_dict, view_purchase_dict):
    total_bought = 0
    total_correct_recommendation = 0
    for key in recommendations_dict:
        recommendation = recommendations_dict[key]
        purchase = view_purchase_dict['test'][key]['purchased']
        total_bought += len(purchase)
        correct_recommendation = [x for x in recommendation if x in purchase]
        total_correct_recommendation += len(correct_recommendation)
    percentage_accuracy = total_correct_recommendation*100/total_bought
    return percentage_accuracy

def create_groupwise_purchase_report(recommendation_dict, view_purchase_dict):
    purchased_in_training = []
    viewed_in_training = []
    not_in_training = []
    for key, items in recommendation_dict.items():
        train_purchase = view_purchase_dict['train'][key]['purchased']
        train_viewed = view_purchase_dict['train'][key]['viewed']
        test_purchase = view_purchase_dict['test'][key]['purchased']
        for item in items:
            if item in test_purchase:
                if item in train_purchase:
                    purchased_in_training.append(item)
                elif item in train_viewed:
                    viewed_in_training.append(item)
                else:
                    not_in_training.append(item)
    return purchased_in_training, viewed_in_training, not_in_training

def calculate_percentage_accuracy_multiple_test_purchase(recommendations_dict, view_purchase_dict):
    total_bought = 0
    total_correct_recommendation = 0
    for key in recommendations_dict:
        recommendation = recommendations_dict[key]
        purchase = view_purchase_dict['test'][key]['purchased']
        total_bought += len(purchase)
        correct_recommendation = [x for x in purchase if x in recommendation]
        total_correct_recommendation += len(correct_recommendation)
    percentage_accuracy = total_correct_recommendation*100/total_bought
    return percentage_accuracy

def create_groupwise_purchase_report_multiple_test_purchase(recommendation_dict, view_purchase_dict):
    purchased_in_training = []
    viewed_in_training = []
    not_in_training = []
    for key, recom_item in recommendation_dict.items():
        train_purchase = view_purchase_dict['train'][key]['purchased']
        train_viewed = view_purchase_dict['train'][key]['viewed']
        test_purchase = view_purchase_dict['test'][key]['purchased']
        for item in test_purchase:
            if item in recom_item:
                if item in train_purchase:
                    purchased_in_training.append(item)
                elif item in train_viewed:
                    viewed_in_training.append(item)
                else:
                    not_in_training.append(item)
    return purchased_in_training, viewed_in_training, not_in_training



def test_purchase_status_in_training_data(train, test):
    viewed = []
    purchased = []
    unseen = []
    view_purchase_dict = item_viewed_purchased(train, test)
    for key in view_purchase_dict['test']:
        if key in view_purchase_dict['train']:
            for item in view_purchase_dict['test'][key]['purchased']:
                if item in view_purchase_dict['train'][key]['purchased']:
                    purchased.append(item)
                elif item in view_purchase_dict['train'][key]['viewed']:
                    viewed.append(item)
                else:
                    unseen.append(item)
    return unseen, viewed, purchased

def accuracy_based_on_coupon_status_in_train_data(train, test, test_user_recommendation_dict, view_purchase_dict):
    unseen, viewed, purchased = test_purchase_status_in_training_data(train, test)
    purchased_in_training, viewed_in_training, not_in_training = create_groupwise_purchase_report_multiple_test_purchase(test_user_recommendation_dict, view_purchase_dict)
    per_accuracy_unseen = len(not_in_training)*100/len(unseen)
    per_accuracy_viewed = len(viewed_in_training)*100/len(viewed)
    per_accuracy_purchased = len(purchased_in_training)*100/len(purchased)
    y1 = [len(unseen), len(viewed), len(purchased)]
    y2 = [len(not_in_training), len(viewed_in_training), len(purchased_in_training)]
    
    test_purchase_break_up = {'Puchased in Training': len(purchased),
                              'Viewed in Training': len(viewed),
                              'Neither Viewed nor Purchased in Training': len(unseen)}
    successful_recommendation_break_up = {'Purchased in Training': len(purchased_in_training),
                                          'Viewed in Training': len(viewed_in_training),
                                          'Neither Viewed nor Purchased in Training': len(not_in_training)}
    per_accuracy_overall = np.sum(y2)*100/np.sum(y1)
    accuracy_report = {'Overall Accuracy': per_accuracy_overall,
                       'Purchased': per_accuracy_purchased, 'Seen': per_accuracy_viewed,
                       'Neither Seen nor purchased': per_accuracy_unseen}
    return accuracy_report, test_purchase_break_up, successful_recommendation_break_up
    
    


def plot_groupwise_purchase_accuracy(train, test, test_user_recommendation_dict, view_purchase_dict):
    unseen, viewed, purchased = test_purchase_status_in_training_data(train, test)
#    purchased_in_training, viewed_in_training, not_in_training = create_groupwise_purchase_report(test_user_recommendation_dict, view_purchase_dict)
    purchased_in_training, viewed_in_training, not_in_training = create_groupwise_purchase_report_multiple_test_purchase(test_user_recommendation_dict, view_purchase_dict)
    per_accuracy_unseen = len(not_in_training)*100/len(unseen)
    per_accuracy_viewed = len(viewed_in_training)*100/len(viewed)
    per_accuracy_purchased = len(purchased_in_training)*100/len(purchased)
    
    bar_width = 0.25
    index = np.arange(3)
    y1 = [len(unseen), len(viewed), len(purchased)]
    y2 = [len(not_in_training), len(viewed_in_training), len(purchased_in_training)]
    plt.bar(index, y1, bar_width, label = 'purchased coupons in the test data') 
    plt.bar(index+0.25, y2, bar_width, label = 'recommended coupons in the test data purchase')
    plt.xticks(index, ('unseen', 'viewed only', 'purchased'))
    plt.ylabel('Number of  Coupons') 
    plt.legend() 
    plt.show()
    
    y = [per_accuracy_unseen, per_accuracy_viewed, per_accuracy_purchased]
    plt.bar(index, y, bar_width, color = 'r')
    plt.xticks(index, ('unseen', 'viewed only', 'purchased')) 
    plt.ylabel('Percentage Accuracy')
    plt.xlim((-0.5, 2.75))
    plt.show()
    

def accuracy_based_on_number_of_purchases_in_test(view_purchase_dict, users_recommendation, test):

    test_users = test.USER_ID_hash.unique().tolist()
    
    df = pd.DataFrame(columns = ['n_purchased_test', 'n_purchase_training', 'n_correct_recommendation'])
    for user in test_users:
        purchased_coupons_training = view_purchase_dict['train'][user]['purchased']
        purchased_coupons_test = view_purchase_dict['test'][user]['purchased']
        recommended_coupons = users_recommendation[user]  
        correct_reco = [coupon for coupon in purchased_coupons_test  if coupon in recommended_coupons]
#        correct_reco = [coupon for coupon in recommended_coupons if coupon in purchased_coupons_test] # older  definition
        n_purchased_training = len(purchased_coupons_training)
        n_purchased_test = len(purchased_coupons_test)
        n_correct = len(correct_reco)
        df.loc[user] =  [n_purchased_test, n_purchased_training, n_correct]
    
    df_1 = df.groupby(by = 'n_purchase_training', as_index = True).sum()        
    df_2 = df.n_purchase_training.value_counts()
    df_merged = pd.concat([df_1, df_2], axis = 1)
    df_merged['per_accuracy'] = df_merged['n_correct_recommendation']*100/df_merged['n_purchased_test']
    return df_merged


def plot_purchase_accuracy_for_different_number_of_users(df_merged):
    bar_width = 0.25
    index = df_merged.index
    max_index  = np.max(df_merged.index.tolist())
    if max_index == 5:
        xticks_loc = range(5)
    else:
        xticks_loc = range(0, 70, 5)
           
    y1 = df_merged.n_purchased_test
    y2 = df_merged.n_correct_recommendation
    plt.bar(index, y1, bar_width, label = 'purchased coupons in the test data') 
    plt.bar(index+0.25, y2, bar_width, label = 'recommended coupons in the test data purchase')
    plt.xticks(xticks_loc, xticks_loc)
    plt.xlabel('Number of coupons bought in the training data by users')
    plt.ylabel('Number of  Coupons') 
    plt.legend() 
    plt.show()
    
    if max_index == 5:
        x_lim = [-0.5, 5.5]
    else:
        x_lim = [-0.5, 65]
        
    bar_width = 0.50    
    y = df_merged.per_accuracy
    plt.bar(index, y, bar_width, color = 'r')
    plt.xticks(xticks_loc, xticks_loc)
    plt.xlabel('Number of coupons bought in the training data by users')
    plt.ylabel('Percentage Accuracy')
    plt.xlim(x_lim)
    plt.show()
    df_merged.loc[:, ['n_purchased_test', 'per_accuracy']]
    return



def get_input_data_details(train, test):
    users = list(set(test.USER_ID_hash.unique().tolist() + train.USER_ID_hash.unique().tolist()))
    n_users =  len(users)
    test_users = test.USER_ID_hash.unique().tolist()
    train_users = train.USER_ID_hash.unique().tolist()
    n_rows =  train.shape[0] + test.shape[0]

    n_coupons_train = len(train.VIEW_COUPON_ID_hash.unique())
    n_users_train = len(train.USER_ID_hash.unique())
    n_users_test = len(test.USER_ID_hash.unique())
    n_rows_test = test.shape[0]
    n_ratings = train.drop_duplicates(subset = ['USER_ID_hash', 'VIEW_COUPON_ID_hash']).shape[0]
    na_perc_in_rating_matrix = 1 - (n_ratings/(n_coupons_train*n_users_train))
    test_users_not_in_training = [user for user in test_users if user not in train_users]
    n_test_users_not_in_training = len(test_users_not_in_training)
    input_data_summary = {'no of users': n_users, 'no of test users': n_users_test,
                          'no of training users': n_users_train,
                          'total number of  rows': n_rows,
                          'no of rows in the test data': n_rows_test,
                          'no of coupons in training': n_coupons_train, 
                          'NA percentage in the rating matrix': na_perc_in_rating_matrix,
                          'no of test users not in training': n_test_users_not_in_training
                          }
    return input_data_summary


    
if __name__ == '__main__':
    coupon_visit = dl.load_coupon_visit_data()
    coupon_visit_subset = sd.create_data_subset(n_users = 1000, min_purchase = 1, max_purchase = 50000, seed_value = 10)
    coupon_clust_visit = dpre.substitute_coupon_id_with_cluster_id(coupon_visit_subset)
    train, test = cttd.create_train_test_set(coupon_clust_visit, train_frac = 0.7, seed_value = 10)
    os.remove('intermediate_result/colf_recommendation_dict.pkl')
    colf_users_recommendation = colf.get_recommendation_for_test_data(train, test)
    view_purchase_dict = item_viewed_purchased(train, test)
    per_accuracy_colf =  calculate_percentage_accuracy(colf_users_recommendation, view_purchase_dict)
    unseen, viewed, purchased = test_purchase_status_in_training_data(train, test)
    print len(unseen), len(viewed), len(purchased)
    purchased_in_training, viewed_in_training, not_in_training = create_groupwise_purchase_report(colf_users_recommendation, view_purchase_dict)
    print len(purchased_in_training), len(viewed_in_training), len(not_in_training)




    
    
    
    
    
    
    
    
    








 
 
    
    
    
    
    


 



#plt.title('Status of purchased coupons during the \n test period in the traing data') 

    


    
    
    
    



# need to check
def prediction_item_comparison(cf_recommendation_dict, conf_recommendation_dict, purchase_dict):
    common = []
    only_cf = []
    only_conf = []
    none = []
    for key in purchase_dict['test']:
        for item in purchase_dict['test'][key]:
            if item in cf_recommendation_dict[key]:
                if item in conf_recommendation_dict[key]:
                    common.append(item)
                if item not in conf_recommendation_dict[key]:
                    only_cf.append(item)
            if item in conf_recommendation_dict[key]:
                if item not in cf_recommendation_dict[key]:
                    only_conf.append(item)
            if item not in cf_recommendation_dict[key]:
                if item not in conf_recommendation_dict[key]:
                    none.append(item)
    return common, only_cf, only_conf, none

if __name__ == "__main__":
    seed_value = random.choice(range(9999))
    coupon_visit_subset = sd.create_data_subset(n_users = 5, min_purchase = 1, max_purchase = 5, seed_value = seed_value)
    coupon_clust_visit = dpre.substitute_coupon_id_with_cluster_id(coupon_visit_subset)
    train, test = cttd.create_train_test_set(coupon_clust_visit, train_frac = 0.7, seed_value = 10)
    test_user_recommendation_dict = colf.get_collaborative_filtering_recommendation(train, test)
    view_purchase_dict = item_viewed_purchased(train, test)
    test_users = test.USER_ID_hash.unique()
    for user in test_users:
        print "user_id:", user
        print "user purchase:", view_purchase_dict['test'][user]['purchased']
        print "user recommendation:", test_user_recommendation_dict[user]
        print "\n\n"
        
    per_accuracy = calculate_percentage_accuracy(test_user_recommendation_dict, view_purchase_dict)
    print "percentage accuracy:", per_accuracy
    unseen, viewed, purchased = test_purchase_status_in_training_data(train, test)
    print len(unseen), len(viewed), len(purchased)
    purchased_in_training, viewed_in_training, not_in_training = create_groupwise_purchase_report(test_user_recommendation_dict, view_purchase_dict)
    print len(purchased_in_training), len(viewed_in_training), len(not_in_training)
    
    



