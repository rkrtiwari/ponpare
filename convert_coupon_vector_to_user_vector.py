# -*- coding: utf-8 -*-
"""
Created on Mon May 15 08:04:13 2017

@author: ravitiwari
"""

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


###############################################################################
# data frame used in the calculation
# visit_user_coupon_detail
###############################################################################

columns_to_use = ['PURCHASE_FLG', 'SEX_ID', 'AGE', 'GENRE_NAME', 'PRICE_RATE',
                  'CATALOG_PRICE']
user_coupon_purchase_detail = visit_user_coupon_detail[columns_to_use]
user_coupon_purchase_detail.dropna(axis = 0, how = 'any', inplace = True)


###############################################################################
# conversion into categorical variables
###############################################################################
# 1.age
X_cat = pd.DataFrame(columns = ['AGE', 'PRICE_RATE', 'CATALOG_PRICE', 'SEX_ID',
                                'GENRE_NAME'])
bins = [0,20,30,40,50,60,100]
sufs = np.arange(len(bins)-1)
labels = ["age" + str(suf) for suf in sufs]
X_cat['AGE'] = pd.cut(user_coupon_purchase_detail.AGE, bins = bins, labels = labels)

# 2. price rate
bins = [-1,25,50,60,70,80,90,100]
sufs = np.arange(len(bins)-1)
labels = ["price_rate" + str(suf) for suf in sufs]
X_cat['PRICE_RATE'] = pd.cut(user_coupon_purchase_detail.PRICE_RATE, bins = bins, labels = labels)

# 3. catalog price
bins = [0, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 1000000]
sufs = np.arange(len(bins)-1)
labels = ["catalog_price" + str(suf) for suf in sufs]
X_cat['CATALOG_PRICE'] = pd.cut(user_coupon_purchase_detail.CATALOG_PRICE, bins = bins, labels = labels)

# 4. sex_id
X_cat['SEX_ID'] = user_coupon_purchase_detail.SEX_ID

# 5. genre_name
X_cat['GENRE_NAME'] = user_coupon_purchase_detail.GENRE_NAME

# 6. predicated variable 
y = user_coupon_purchase_detail.PURCHASE_FLG



###############################################################################
# what are the coupons that are availbale
###############################################################################
X_cat.head()
y.head()

###############################################################################
# convert user content into user vector
###############################################################################
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

# test the code 
# a. single user    
user_vector_from_user_content('age4', 'f')    

# b. multiple user
n_users = 100
uv = np.zeros((n_users, 8))
for i in xrange(n_users):
    age_cat = X_cat.iloc[i,0]
    sex_cat = X_cat.iloc[i,3]
    uv[i] = user_vector_from_user_content(age_cat, sex_cat)  


###############################################################################
# conditional probability calculation
###############################################################################
u_features = ["AGE", "SEX_ID"]                             # u: user
c_features = ["GENRE_NAME", "PRICE_RATE", "CATALOG_PRICE"] # c: coupon

coupon_cond_prob = pd.DataFrame(columns = ('coupon_feature', 'user_feature', 
                                           'coupon_feature_value', 'user_feature_value',
                                           'cond_prob'))
i = 0
for c_feature in c_features:
    c_feature_values = X_cat[c_feature].unique()
    c_value_count =  X_cat[c_feature].value_counts()
    c_total = sum(c_value_count)
    for c_feature_value in c_feature_values:
        c_prob =  c_value_count.loc[c_feature_value]/c_total
        for u_feature in u_features:
            u_feature_values = X_cat[u_feature].unique()
            u_value_count =  X_cat[u_feature].value_counts()
            u_total = sum(u_value_count)
            
            ind = X_cat[c_feature] == c_feature_value
            u_feature_value_cond = X_cat[ind][u_feature].unique()
            u_value_count_cond =  X_cat[ind][u_feature].value_counts()
            u_total_cond = sum(u_value_count_cond)
            for u_feature_value in u_feature_values:
                u_prob =  u_value_count.loc[u_feature_value]/u_total
                u_prob_cond = u_value_count_cond.loc[u_feature_value]/u_total_cond
                post_prob = c_prob*u_prob_cond/u_prob
                coupon_cond_prob.loc[i] = [c_feature, u_feature, c_feature_value,
                                    u_feature_value, post_prob]
                i += 1
                
coupon_cond_prob.sort_values(by = ['coupon_feature', 'user_feature', 
                                   'coupon_feature_value', 'user_feature_value'], 
    inplace = True)


##########################################################################
# coupon features to user vector
#########################################################################

# function to convert coupon feature to user vector
def coupon_feature_to_user_vector(coupon_feature_names, user_features, coupon_cond_prob = coupon_cond_prob):
    coupon_user_content = np.zeros(8)
    w = [1,1,1]
    for cf_name, u_f in zip(coupon_feature_names,user_features):
        ind1 = coupon_cond_prob.coupon_feature_value == cf_name
        ind2 = coupon_cond_prob.user_feature == u_f
        ind = ind1 & ind2
        df = coupon_cond_prob.loc[ind]
        
        if u_f == 'SEX_ID':
            u_v = df.cond_prob.values
            fill_array = np.zeros(6)
            uv_full = np.concatenate((u_v, fill_array))
            coupon_user_content += w[0]*uv_full
            
        else:
            u_v = df.cond_prob.values
            fill_array = np.zeros(2)
            uv_full = np.concatenate((fill_array, u_v))
            coupon_user_content += uv_full
            
    return coupon_user_content


# results            
GENRE_NAME =  'ホテル・旅館'
CATALOG_PRICE = 'catalog_price7'
PRICE_RATE = 'price_rate1'

coupon_feature_names = ['ホテル・旅館', 'catalog_price7', 'price_rate1']
user_features = ['SEX_ID','AGE', 'AGE']            
            
coupon_feature_to_user_vector(coupon_feature_names, user_features, coupon_cond_prob)            
            
            
