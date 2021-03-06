# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 09:19:08 2017

@author: tiwarir
"""
from __future__ import division
import subset_data as sd
import numpy as np
import os
import pickle
import random
import data_loading as dl

def create_train_test_set(df, train_frac = 0.7, seed_value = 10):
    n_users = len(df.USER_ID_hash.unique())
    fname_train = "intermediate_result/train_" + "user_"+ str(n_users) + "_seed_" + str(seed_value) + ".pkl"
    fname_test = "intermediate_result/test_" + "user_"+ str(n_users) + "_seed_" + str(seed_value) + ".pkl"
    if os.path.isfile(fname_train):
        train = pickle.load(open(fname_train,'rb'))
        test = pickle.load(open(fname_test,'rb'))
    else:
        np.random.seed(100)
        n_obs = len(df)
        ind_train = np.random.choice(n_obs, size = int(train_frac*n_obs), replace = False)
        ind_test = [x for x in range(n_obs) if x not in ind_train]
        train = df.iloc[ind_train]
        test = df.iloc[ind_test]
        pickle.dump(train, open(fname_train, 'wb '))
        pickle.dump(test, open(fname_test, 'wb '))
    return train, test


if __name__ == '__main__': 
    n_users = 10
    seed_value = random.choice(range(1000000))  
    df = sd.create_data_subset(n_users = n_users, min_purchase = 1, max_purchase = 20000, seed_value = seed_value)
    
    users = df.USER_ID_hash.unique().tolist()
    coupon_visit = dl.load_coupon_visit_data()
    ind = coupon_visit.USER_ID_hash.isin(users)
    n_users = len(users)
    fname_train = "intermediate_result/train_" + "user_"+ str(n_users) + "_seed_" + str(seed_value) + ".pkl"
    fname_test = "intermediate_result/test_" + "user_"+ str(n_users) + "_seed_" + str(seed_value) + ".pkl"
    fnames = [fname_train, fname_test]
    
    for fname in fnames:
        if os.path.isfile(fname):
            os.remove(fname)
#            print "removed: ", fname
    
    
    train, test = create_train_test_set(df, train_frac = 0.7, seed_value = seed_value)
    df_users = df.USER_ID_hash.unique().tolist()
    train_users = train.USER_ID_hash.unique().tolist()
    test_users = test.USER_ID_hash.unique().tolist()
    train_test_users = list(set(train_users + test_users))
    
    print "subset data shape:", df.shape
    print "train, test shape:", train.shape, test.shape
    print "\n"
    print "rows in subset data:", df.shape[0]
    print "sum of rows in train and test:", train.shape[0] + test.shape[0]
    print "number of rows for chosen users in the original view/purchase data:", np.sum(ind)
    print "\n"
    train_per = train.shape[0]/(train.shape[0] + test.shape[0])
    print "train rows percentage:", train_per*100
    test_per = test.shape[0]/(train.shape[0] + test.shape[0])
    print "test rows percentage:", test_per*100
    print "train + test percentage:", (train_per + test_per)*100
    print "\n"
    print "number of unique users in train, test data:", len(train_users), len(test_users)
    print "unique users in train, test data combined:", len(train_test_users)
    print "unique users in the subset data:", len(df_users)
    
    
    
    

