# -*- coding: utf-8 -*-
"""
Created on Thu May 04 14:08:50 2017

@author: ravitiwari
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

###############################################################################
# existing data frames
# 1. user_list
# 2. coupon_list_train
# 3. coupon_detail_train  : purchase data
# 4. coupon_visit_train
# 5. coupon_area_train
###############################################################################

###############################################################################
# merging of dataframes
# coupon visit data is merged with user detail and coupon detail
###############################################################################
# 1. removed duplicated columns so that there is only one entry for a given combination
# of user_id and coupon_id. Either the user has bought a given coupon or he has not 
# bought
coupon_visit_train.sort_values(by = ['USER_ID_hash','VIEW_COUPON_ID_hash', 'PURCHASE_FLG'], inplace = True)
coupon_visit_train.drop_duplicates(subset = ['USER_ID_hash','VIEW_COUPON_ID_hash'], keep = "last", inplace = True)

# 2. remove unnecessary columns
columns_to_keep = ["PURCHASE_FLG", "USER_ID_hash", "VIEW_COUPON_ID_hash"]
coupon_visit = coupon_visit_train[columns_to_keep]

# 3. merge to get user and coupon detail
visit_user_detail = coupon_visit.merge(user_list, on = 'USER_ID_hash', how = 'left')
visit_user_coupon_detail = visit_user_detail.merge(coupon_list_train, left_on = 'VIEW_COUPON_ID_hash',
                                                   right_on = 'COUPON_ID_hash', how = 'left')

# 4. remove unnecessary columns
columns_to_keep = ['PURCHASE_FLG', 'USER_ID_hash', 'SEX_ID', 'AGE', 'PREF_NAME',
                   'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE',
                   'VALIDFROM', 'VALIDEND', 'ken_name', 'COUPON_ID_hash', 'VIEW_COUPON_ID_hash']
visit_user_coupon_detail = visit_user_coupon_detail[columns_to_keep]










