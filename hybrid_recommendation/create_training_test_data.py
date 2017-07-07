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
    df = sd.create_data_subset(n_users = 100, min_purchase = 1, max_purchase = 20000, seed_value = 10)
    train, test = create_train_test_set(df, train_frac = 0.7, seed_value = 10)
    

