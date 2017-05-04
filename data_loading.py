# -*- coding: utf-8 -*-
"""
Created on Thu May 04 12:29:04 2017

@author: ravitiwari
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


###############################################################################
# setting the directory
###############################################################################
os.getcwd()
new_dir = os.path.join("Documents", "ponpare")
os.chdir(new_dir)
os.getcwd()
files = os.listdir(".")

###############################################################################
# creating directory if it does not exists (RUN IT ONLY ONCE)
###############################################################################
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

###############################################################################
# unzipping the file (RUN IT ONLY ONCE)
###############################################################################
files = os.listdir(".")

for f in files:
    if f.endswith(".zip"):
        zip_ref = zipfile.ZipFile(f, 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()
        
###############################################################################        
# reading the data
###############################################################################
# 1. user list
fname = 'user_list.csv'
fname = os.path.join("data", fname)
user_list = pd.read_csv(fname)

# 2. coupon visit train
fname = 'coupon_visit_train.csv' 
fname = os.path.join("data", fname)
coupon_visit_train = pd.read_csv(fname)

# 3. coupon list train
fname = 'coupon_list_train.csv' 
fname = os.path.join("data", fname)
coupon_list_train = pd.read_csv(fname)


# 4. coupon area train
fname = 'coupon_area_train.csv' 
fname = os.path.join("data", fname)
coupon_area_train = pd.read_csv(fname)

# 5. coupon detail train
fname = 'coupon_detail_train.csv'
fname = os.path.join("data", fname)
coupon_detail_train = pd.read_csv(fname)

###############################################################################
# understanding data size
###############################################################################
n_users = user_list.shape
n_coupons = coupon_list_train.shape
n_visits = coupon_visit_train.shape
n_purchases = coupon_detail_train.shape








