# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:36:42 2017

@author: tiwarir
"""

ind1 = coupon_cond_prob.coupon_feature == "GENRE_NAME"
ind2 = coupon_cond_prob.user_feature == "SEX_ID"
ind = ind1 & ind2
coupon_cond_prob[ind]

###############################################################################
# convert a given gender and genre into content vector
###############################################################################
# columns names
X_encoded.columns
coupon_cond_prob.columns
coupon_cond_prob.coupon_feature.unique()
coupon_cond_prob.user_feature.unique()
coupon_cond_prob.coupon_feature_value.unique()
# user columns
# SEX_ID_f, SEX_ID_m
# AGE_age0, AGE_age1, AGE_age2, AGE_age3, AGE_age4, AGE_5


# 1. GENRE_NAME to SEX_ID

GENRE_NAME =  'ホテル・旅館'
user_feature = 'SEX_ID'

ind1 = coupon_cond_prob.coupon_feature_value == GENRE_NAME
ind2 = coupon_cond_prob.user_feature == user_feature
ind = ind1 & ind2
df = coupon_cond_prob.loc[ind]
df.columns
w = [1,1,1]
coupon_user_content = np.zeros(8)
prob_val = df.cond_prob.values
fill_array = np.zeros(6)
sex_id_array = np.concatenate((prob_val, fill_array))

# 2. 
CATALOG_PRICE = 'catalog_price7'
user_feature = 'AGE'
ind1 = coupon_cond_prob.coupon_feature_value == CATALOG_PRICE
ind2 = coupon_cond_prob.user_feature == user_feature
ind = ind1 & ind2
coupon_cond_prob.loc[ind]

# 3
PRICE_RATE = 'price_rate1'
user_feature = 'AGE'
ind1 = coupon_cond_prob.coupon_feature_value == PRICE_RATE
ind2 = coupon_cond_prob.user_feature == user_feature
ind = ind1 & ind2
coupon_cond_prob.loc[ind]


##########################################################################
# coupon features
#########################################################################
coupon_cond_prob = coupon_cond_prob.sort_values(by = ['coupon_feature','user_feature',
                                'coupon_feature_value','user_feature_value'])

GENRE_NAME =  'ホテル・旅館'
CATALOG_PRICE = 'catalog_price7'
PRICE_RATE = 'price_rate1'

coupon_feature_names = ['ホテル・旅館', 'catalog_price7', 'price_rate1']
user_features = ['SEX_ID','AGE', 'AGE']

w = [1,1,1]
coupon_user_content = np.zeros(8)
for cf_name, u_f in zip(coupon_feature_names,user_features):
    print cf_name, u_f
    ind1 = coupon_cond_prob.coupon_feature_value == cf_name
    ind2 = coupon_cond_prob.user_feature == u_f
    ind = ind1 & ind2
    df = coupon_cond_prob.loc[ind]
    if u_f == 'SEX_ID':
        u_v = df.cond_prob.values
        fill_array = np.zeros(6)
        uv_full = np.concatenate((u_v, fill_array))
        coupon_user_content += uv_full
    else:
        u_v = df.cond_prob.values
        fill_array = np.zeros(2)
        uv_full = np.concatenate((fill_array, u_v))
        coupon_user_content += uv_full
    
    

    

    
    










