# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:36:42 2017

@author: tiwarir
"""
###############################################################################
# what are the coupons that are availbale
###############################################################################
X_cat.head()
y.head()
uv = np.zeros((5,8))
for i in xrange(5):
    age_cat = X_cat.iloc[i,0]
    a_v = np.zeros(6)
    ind = int(age_cat[3])
    a_v[ind] = 1
    sex_cat = X_cat.iloc[i,3]
    if sex_cat == 'm':
        s_v = np.array([1,0])
    else:
        s_v = np.array([0,1])
    u_v = np.concatenate([s_v,a_v])
    uv[i] = u_v
    print u_v




###############################################################################
# user features to user vector
###############################################################################
X_cat.iloc[1]

##########################################################################
# coupon features to user vector
#########################################################################
coupon_cond_prob = coupon_cond_prob.sort_values(by = ['coupon_feature','user_feature',
                                'coupon_feature_value','user_feature_value'])

GENRE_NAME =  'ホテル・旅館'
CATALOG_PRICE = 'catalog_price7'
PRICE_RATE = 'price_rate1'

coupon_feature_names = ['ホテル・旅館', 'catalog_price7', 'price_rate1']
user_features = ['SEX_ID','AGE', 'AGE']

w = [2,1,1]
coupon_user_content = np.zeros(8)
for cf_name, u_f in zip(coupon_feature_names,user_features):
    print cf_name, u_f
    ind1 = coupon_cond_prob.coupon_feature_value == cf_name
    ind2 = coupon_cond_prob.user_feature == u_f
    ind = ind1 & ind2
    df = coupon_cond_prob.loc[ind]
    if u_f == 'SEX_ID':
        u_v = df.cond_prob.values
        print u_v
        fill_array = np.zeros(6)
        uv_full = np.concatenate((u_v, fill_array))
#        coupon_user_content += np.multiply(uv_full,w[0])
        coupon_user_content += w[0]*uv_full
    else:
        u_v = df.cond_prob.values
        print u_v
        fill_array = np.zeros(2)
        uv_full = np.concatenate((fill_array, u_v))
        coupon_user_content += uv_full
        
coupon_user_content    



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



    

    

    
    









