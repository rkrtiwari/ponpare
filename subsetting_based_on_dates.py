# -*- coding: utf-8 -*-
"""
Created on Thu May 04 16:42:49 2017

@author: ravitiwari
"""

from datetime import timedelta

###############################################################################
# make the subset of the coupons based on training
###############################################################################
coupon_list_train.head()
coupon_list_train.DISPFROM = pd.to_datetime(coupon_list_train.DISPFROM)
coupon_list_train.sort_values(by = "DISPFROM", inplace = True)
coupon_list_train.DISPFROM.max()
coupon_list_train.DISPFROM.min()
coupon_list_train.DISPFROM.head()
coupon_list_train.DISPFROM.tail()

np.sum(coupon_list_train.VALIDFROM.isnull())
len(coupon_list_train.VALIDFROM)

cut_off_date = coupon_list_train.DISPFROM.max() - timedelta(days = 7)
ind = coupon_list_train.DISPFROM < cut_off_date

train = coupon_list_train.COUPON_ID_hash[ind]
test = coupon_list_train.COUPON_ID_hash[~ind]

coupon_list_train.DISPFROM.max() + timedelta(days = 7)
pd.to_datetime(coupon_list_train.DISPFROM.max()) - timedelta(days = 7)
type(coupon_list_train.DISPFROM.max())


###############################################################################
# checking the coupon training data to see the appearance of same coupon appearing
# in two different locations
#################################################################################

coupon_list_train.columns
coupon_list_train.shape
coupon_list_train.size

len(coupon_list_train.COUPON_ID_hash.unique())
coupon_list_train.COUPON_ID_hash.value_counts()

###############################################################################
# coupon visit train
###############################################################################

fname = 'coupon_visit_train.csv' 
fname = os.path.join("data", fname)
coupon_visit_train = pd.read_csv(fname)

coupon_visit_train.head()
coupon_visit_train.I_DATE = pd.to_datetime(coupon_visit_train.I_DATE)
coupon_visit_train.I_DATE


coupon_visit_train.columns
coupon_visit_train = coupon_visit_train.sort_values(by = ['VIEW_COUPON_ID_hash', 'I_DATE'])
coupon_vist_train_first_view = coupon_visit_train.drop_duplicates(subset = 'VIEW_COUPON_ID_hash', keep = 'first')
len(coupon_vist_train_first_view)
len(coupon_vist_train_first_view.VIEW_COUPON_ID_hash.unique())





