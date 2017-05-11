# -*- coding: utf-8 -*-
"""
Created on Tue May 09 13:39:18 2017

@author: ravitiwari
"""
from collections import defaultdict
from __future__ import division

###############################################################################
# get data for only purchased coupons
###############################################################################
coupon_visit_train.head()
pur_ind = coupon_visit_train.PURCHASE_FLG == 1
purchased_coupons = coupon_visit_train[pur_ind]






##############################################################################
# merging purchased coupon data with user and coupon information
##############################################################################
purchased_coupons.head()
columns_to_keep = ['I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash']
purchased_coupons = purchased_coupons[columns_to_keep]
purchased_coupon_user_info = purchased_coupons.merge(user_list, how = 'left', 
                                                     on = 'USER_ID_hash')

purchased_user_coupon_info = purchased_coupon_user_info.merge(coupon_list_train,
                how = 'left', left_on = 'VIEW_COUPON_ID_hash', right_on = 'COUPON_ID_hash')

purchased_user_coupon_info.shape
purchased_user_coupon_info.head()

columns_to_keep = ['I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash', 'PREF_NAME',
                   'ken_name']
purchased_user_coupon = purchased_user_coupon_info[columns_to_keep]


###############################################################################
#
###############################################################################
purchased_coupon_ken_by_user_pref = pd.DataFrame(columns = ['pref', 'ken',
                                                         'count', 'per_purchase'])

coupon_kens = purchased_user_coupon.ken_name.unique()
user_prefs = purchased_user_coupon.PREF_NAME.unique()
for pref in user_prefs:
    ind = purchased_user_coupon.PREF_NAME == pref
    df = purchased_user_coupon.loc[ind]
    df_value_counts = df.ken_name.value_counts()
    n_entry = len(df_value_counts)
    pref_s = pd.Series(index=range(n_entry))
    for i in range(n_entry):
        pref_s.loc[i] = pref
    ken = pd.Series(df_value_counts.index)
    count = pd.Series(df_value_counts.values)
    per = count*100/np.sum(count)
    df_2 = pd.DataFrame({'pref':pref_s, 'ken': ken, 'count': count, 'per_purchase': per })
    purchased_coupon_ken_by_user_pref = pd.concat([purchased_coupon_ken_by_user_pref, df_2], ignore_index=True)
    
    
purchased_coupon_ken_by_user_pref.sort_values(by = ['pref', 'count'], 
                                              inplace = True, ascending = False)    
    
purchased_coupon_ken_by_user_pref.iloc[0:100]

# testing of the algorithm. sum of the all perchase should be 100
i = 2
for pref in user_prefs:
    ind = purchased_coupon_ken_by_user_pref.pref == pref
    df = purchased_coupon_ken_by_user_pref.loc[ind]
    per_sum = np.sum(df.per_purchase)
    ind1 = purchased_user_coupon_info.PREF_NAME == pref
    ind2 = purchased_user_coupon_info.ken_name == df.iloc[i,1]
    ind = ind1 & ind2
    cal_sum = np.sum(ind)
    print "pref: ", pref, "ken_name: ",df.iloc[i,1], "purchase_count: ", df.iloc[i,0], "cal: ", cal_sum #, per_sum
    



###############################################################################
# conditional probability
# things to note
# the user area is PREF_NAME
# the coupon area is ken name
###############################################################################
purchased_user_coupon.dropna(axis=0, how = 'any', inplace = True)
(107753 - 84599)/107753.0 # around 20% missing values
coupon_ken_prob_cond_user_pref = pd.DataFrame(columns = ('coupon_ken','user_pref', 
                                'ken_prob', 'pref_prob', 'ken_prob_cond_pref'))

purchased_user_coupon.shape

coupon_kens = purchased_user_coupon.ken_name.unique()
user_prefs = purchased_user_coupon.PREF_NAME.unique()

ken_value_counts = purchased_user_coupon.ken_name.value_counts()
pref_value_counts = purchased_user_coupon.PREF_NAME.value_counts()
ken_total = np.sum(ken_value_counts)
pref_total = np.sum(pref_value_counts)
i = 0
for ken in coupon_kens:
    k_ind = ken_value_counts.index == ken
    ind1 = purchased_user_coupon.ken_name == ken
    ken_prob = ken_value_counts[k_ind]/ken_total
    for pref in user_prefs:
        p_ind = pref_value_counts.index == pref
        pref_prob = pref_value_counts[p_ind]/pref_total 
        if pref_prob[0] == 0.0: print pref_prob #continue
        ind2 = purchased_user_coupon.PREF_NAME == pref
        ind = ind1 & ind2
        pref_prob_cond = np.sum(ind)/np.sum(ind1)
        post_prob = ken_prob[0]*pref_prob_cond/pref_prob[0]
        coupon_ken_prob_cond_user_pref.loc[i] = [ken, pref, ken_prob[0], pref_prob[0],
                                           post_prob]
        i+=1
        
        
coupon_ken_prob_cond_user_pref.columns
coupon_ken_prob_cond_user_pref.sort_values(by = ['pref_prob', 'ken_prob_cond_pref'],
                                           ascending = False, inplace = True)

###############################################################################
# checking the result
###############################################################################

coupon_ken_prob_cond_user_pref["prob_product"] = coupon_ken_prob_cond_user_pref.ken_prob_cond_pref*coupon_ken_prob_cond_user_pref.pref_prob


for ken in coupon_kens:
    ind = coupon_ken_prob_cond_user_pref.coupon_ken == ken
    cal_prob = np.sum(coupon_ken_prob_cond_user_pref.prob_product[ind])
    org_prob = coupon_ken_prob_cond_user_pref[ind]
    print ken, cal_prob, org_prob.ken_prob.iloc[0]

user_prefs
coupon_ken_prob_cond_user_pref
pref_name = '東京都'


# keep only top 5 ken from where users buy
coupon_ken_prob_cond_user_pref.sort_values(by = ['user_pref', 'ken_prob_cond_pref'],
                                           ascending = False, inplace = True)
top_kens_for_pref = pd.DataFrame(columns = ('coupon_ken','user_pref', 
                                'ken_prob', 'pref_prob', 'ken_prob_cond_pref'))
keep_n = 5
pref_name = '東京都'
for pref in user_prefs:
    print pref
    ind = coupon_ken_prob_cond_user_pref.user_pref == pref
    df = coupon_ken_prob_cond_user_pref.loc[ind]
    pd.concat([top_kens_for_pref, df.iloc[:keep_n,]], axis = 0)
#    top_kens_for_pref.append(df.iloc[:keep_n,])
    print df.iloc[:keep_n,]
    
    
    



# test subsetting
np.sum(coupon_visit_train.PURCHASE_FLG)
np.sum(pur_ind)
len(purchased_coupons)
purchased_coupons.shape

# shape before and after merging
purchased_coupons.shape
purchased_coupon_user_info.shape
purchased_user_coupon_info.shape
purchased_coupon_user_info.head()

# check user list
user_list.head()
user_list.shape
len(user_list.USER_ID_hash.unique())

# check coupon list
coupon_list_train.head()
len(coupon_list_train.COUPON_ID_hash.unique())
len(coupon_list_train.COUPON_ID_hash)
coupon_visit_train.head()

# check users purchase
len(purchased_coupons.USER_ID_hash)
len(purchased_coupons.USER_ID_hash.unique())

# checking coupon area

coupon_area_train.head()
coupon_area_train.shape
len(coupon_area_train.COUPON_ID_hash)
len(coupon_area_train.COUPON_ID_hash.unique())
# conclusion same coupon is available in so many areas. Keep that in mind when 
# you refine the algorithm
coupon_pref = coupon_area_train.drop_duplicates(subset = ['PREF_NAME', 'COUPON_ID_hash'],
                                                keep = 'last')

coupon_pref.COUPON_ID_hash[349]
coupon_pref.PREF_NAME[348]

coupon_pref_dict = defaultdict(list)

for i in range(len(coupon_pref.COUPON_ID_hash)):
    print i
    coupon_pref_dict[coupon_pref.COUPON_ID_hash.iloc[i]].append(coupon_pref.PREF_NAME.iloc[i])

len(coupon_pref_dict.keys())
    
for key, value in coupon_pref_dict.items():
    if len(value) > 10:
        print key, value

# i will make use of the above information later but for now, I am leaving out
# this information. what i can do is to out of my recommendation keep only those
# that meet the time as well as the prefecture criteria. First the user is in the 
# prefecture and second the coupon is available in those prefectures where user
# buys coupons from





























visit_user_coupon_detail.head()
visit_user_coupon_detail.dropna(axis = 0, subset = ['PREF_NAME', 'ken_name'], 
                                how  = 'any', inplace = True)
visit_user_coupon_detail.head()
visit_user_coupon_detail.shape

# find the conditional probability
###############################################################################
###############################################################################
###############################################################################
# conditional probability  
###############################################################################
###############################################################################
###############################################################################

u_features = ["PREF_NAME"]                             # u: user
c_features = ["ken_name"] #                            # c: coupon

coupon_cond_prob = pd.DataFrame(columns = ('coupon_feature', 'user_feature', 
                                           'coupon_feature_value', 'user_feature_value',
                                           'cond_prob'))
i = 0
for c_feature in c_features:
    c_feature_values = visit_user_coupon_detail[c_feature].unique()
    c_value_count =  visit_user_coupon_detail[c_feature].value_counts()
    for c_feature_value in c_feature_values:
        c_prob =  c_value_count.loc[c_feature_value]/c_total
        for u_feature in u_features:
            u_feature_values = visit_user_coupon_detail[u_feature].unique()
            u_value_count =  visit_user_coupon_detail[u_feature].value_counts()
            u_total = sum(u_value_count)
            
            ind = visit_user_coupon_detail[c_feature] == c_feature_value
            u_feature_value_cond = visit_user_coupon_detail[ind][u_feature].unique()
            u_value_count_cond =  visit_user_coupon_detail[ind][u_feature].value_counts()
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
                
ind1 = coupon_cond_prob.coupon_feature == "GENRE_NAME"
ind2 = coupon_cond_prob.user_feature == "SEX_ID"
ind = ind1 & ind2
coupon_cond_prob[ind]

coupon_cond_prob.to_csv("conditional_probability.csv", index = False)
