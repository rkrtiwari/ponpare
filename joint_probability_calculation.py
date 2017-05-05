# -*- coding: utf-8 -*-
"""
Created on Thu May 04 16:14:44 2017

@author: ravitiwari
"""

###############################################################################
###############################################################################
# functions to calculate unconditional (marginal) probabilities.
# the probability values are stored in the user_list_prob
###############################################################################
###############################################################################

user_list_prob = pd.DataFrame(columns = ('feature_name', 'feature_value', 'prob'))
df = pd.DataFrame(columns = ('feature_name', 'feature_value', 'prob'))
feature_list = ['SEX_ID', 'PREF_NAME']

for feature in feature_list:
    value_counts  = user_list[feature].value_counts()
    total = sum(value_counts)
    for i in range(len(value_counts)):
        prob = value_counts[i]/total
        df.loc[i]  = [feature, value_counts.index[i], prob]        
    user_list_prob = user_list_prob.append(df, ignore_index = True)


###############################################################################
###############################################################################
###############################################################################
# conditional probability  
###############################################################################
###############################################################################
###############################################################################

u_features = ["AGE", "SEX_ID"]                             # u: user
c_features = ["GENRE_NAME", "PRICE_RATE", "CATALOG_PRICE"] # c: coupon

###############################################################################
# coupon unconditional probability
##############################################################################
for c_feature in c_features:
    c_feature_values = X_cat[c_feature].unique()
    c_value_count =  X_cat[c_feature].value_counts()
    c_total = sum(c_value_count)
    for c_feature_value in c_feature_values:
        c_prob =  c_value_count.loc[c_feature_value]/c_total
        print c_feature, c_feature_value, c_prob, c_total, c_value_count.loc[c_feature_value]

###############################################################################
# user unconditional probability
###############################################################################
for u_feature in u_features:
    u_feature_values = X_cat[u_feature].unique()
    u_value_count =  X_cat[u_feature].value_counts()
    u_total = sum(u_value_count)
    for u_feature_value in u_feature_values:
        u_prob =  u_value_count.loc[u_feature_value]/u_total
        print u_feature, u_feature_value, u_prob

###############################################################################
# conditional probability
###############################################################################
coupon_cond_prob = pd.DataFrame(columns = ('coupon_feature', 'user_feature', 
                                           'coupon_feature_value', 'user_feature_value',
                                           'cond_prob'))
i = 0
for c_feature in c_features:
    c_feature_values = X_cat[c_feature].unique()
    c_value_count =  X_cat[c_feature].value_counts()
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
                
                
