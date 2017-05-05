# -*- coding: utf-8 -*-
"""
Created on Thu May 04 16:14:44 2017

@author: ravitiwari
"""

###############################################################################
# pandas setting
###############################################################################
pd.set_option('expand_frame_repr', False)


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
coupon_prob = pd.DataFrame(columns = ('coupon_feature','coupon_feature_value', 
                                           'prob'))

i = 0
for c_feature in c_features:
    c_feature_values = X_cat[c_feature].unique()
    c_value_count =  X_cat[c_feature].value_counts()
    c_total = sum(c_value_count)
    for c_feature_value in c_feature_values:
        c_prob =  c_value_count.loc[c_feature_value]/c_total
        coupon_prob.loc[i] = [c_feature, c_feature_value, c_prob]
        i += 1

###############################################################################
# user unconditional probability
###############################################################################
user_prob = pd.DataFrame(columns = ('user_feature','user_feature_value', 
                                           'prob'))

i = 0
for u_feature in u_features:
    u_feature_values = X_cat[u_feature].unique()
    u_value_count =  X_cat[u_feature].value_counts()
    u_total = sum(u_value_count)
    for u_feature_value in u_feature_values:
        u_prob =  u_value_count.loc[u_feature_value]/u_total
        user_prob.loc[i] = [u_feature, u_feature_value, u_prob]
        i += 1


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
                
                
ind1 = coupon_cond_prob.coupon_feature == "GENRE_NAME"
ind2 = coupon_cond_prob.user_feature == "SEX_ID"
ind = ind1 & ind2
coupon_cond_prob[ind]

###############################################################################
# module to test conditional probability values
###############################################################################
#sample_data = coupon_cond_prob.sample()
n_row, n_col = coupon_cond_prob.shape
n = np.random.choice(n_row)
sample_data = coupon_cond_prob.loc[22] # 6,7,14,15,22,23

sample_data.coupon_feature
sample_data.user_feature
sample_data.coupon_feature_value
sample_data.user_feature_value
sample_data.cond_prob

# coupon feature probability
coupon_value_count = pd.value_counts(X_cat[sample_data.coupon_feature].values.flatten())
coupon_feature_count = coupon_value_count[sample_data.coupon_feature_value]
coupon_total = np.sum(coupon_value_count)
prob_coupon_feature = coupon_feature_count/coupon_total

# user feature probability
user_value_count = pd.value_counts(X_cat[sample_data.user_feature].values.flatten())
user_feature_count = user_value_count[sample_data.user_feature_value]
user_total = np.sum(user_value_count)
prob_user_feature = user_feature_count/user_total

# likelihood
user_feature_count = user_value_count[sample_data.user_feature_value]
ind1 = X_cat[sample_data.coupon_feature] == sample_data.coupon_feature_value
ind2 = X_cat[sample_data.user_feature] == sample_data.user_feature_value
ind = ind1 & ind2
user_and_coupon_feature_count = np.sum(ind)
likelihood = user_and_coupon_feature_count/coupon_feature_count

posterior = prob_coupon_feature*likelihood/prob_user_feature

###############################################################################
# another module to test conditional probability: need to generalize
###############################################################################
coupon_feature = "グルメ"
user_feature_1 = 'f'
user_feature_2 = 'm'

p_cf_cond_uf_1 = 0.217242
p_cf_cond_uf_2 = 0.281748
p_uf_1 = 0.557716
p_uf_2 = 0.442284

p_cf = 0.245772
p_cf_calc = p_cf_cond_uf_1*p_uf_1 + p_cf_cond_uf_2*p_uf_2



















