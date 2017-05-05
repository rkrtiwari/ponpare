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
# conditional joint probability  
###############################################################################
###############################################################################
###############################################################################

u_features = ["AGE", "SEX_ID"]                             # u: user
c_features = ["GENRE_NAME", "PRICE_RATE", "CATALOG_PRICE"] # c: coupon

for c_feature in c_features:
    c_feature_values = X_cat[c_feature].unique()
    c_value_count =  X_cat[c_feature].value_counts()
    c_total = sum(c_value_count)
    for c_feature_value in c_feature_values:
        c_prob =  c_value_count.loc[c_feature_value]/c_total
        print c_feature, c_feature_value, c_prob, c_total, c_value_count.loc[c_feature_value]


for u_feature in u_features:
    u_feature_values = X_cat[u_feature].unique()
    u_value_count =  X_cat[u_feature].value_counts()
    u_total = sum(u_value_count)
    for u_feature_value in u_feature_values:
        u_prob =  u_value_count.loc[u_feature_value]/u_total
        print u_feature, u_feature_value, u_prob


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
                print c_feature, u_feature, c_feature_value, u_feature_value, post_prob

                
            




    

for cfeature in coupon_feature:
    for ufeature in user_feature:
        print cfeature, ufeature
        print X_cat[ufeature].value_counts()
        print X_cat[cfeature].value_counts()


n_u_features = len(X_cat["GENRE_NAME"].unique())

for i in range(len(X_cat["GENRE_NAME"])):
    u_value_count = X_cat["GENRE_NAME"].value_counts()
    print u_value_count 


    
###############################################################################
################################################################################
###############################################################################
# function to calculate conditional probability
###############################################################################
###############################################################################
###############################################################################
coup_prob = {}

purchase_clist_ulist.columns
col_names = list(purchase_clist_ulist.columns)
ind1 = col_names.index("GENRE_NAME")
ind2 = col_names.index("SMALL_AREA_NAME")
i_columns = [ind1,ind2]
purchase_clist_ulist.columns[i_columns]

## probability generation
for i in i_columns:
    gen_prob = purchase_clist_ulist.iloc[:,i].value_counts()/len(purchase_clist_ulist)
    coup_prob[purchase_clist_ulist.columns[i]] = gen_prob.to_dict()

## printing probabilities
for key in coup_prob.keys():
    for subkey, value in coup_prob[key].items():
        print key, subkey, value





prob = []              
for ccol in coupon_cols:
    c_variables = purchase_clist_ulist[ccol].unique()
    for ucol in user_cols:
        u_variables = purchase_clist_ulist[ucol].unique()
        for c_variable in c_variables:
            ind_c = purchase_clist_ulist[ccol] == c_variable
            for u_variable in u_variables:
                ind_u = purchase_clist_ulist[ind_c][ucol]==u_variable
                print c_variable, u_variable, sum(ind_u)/sum(ind_c)
                prob.append((c_variable, u_variable, sum(ind_u)/sum(ind_c)))
        
my_dict = {}
my_dict[('SEX_ID', 'GENRE_NAME')] = [('a','b',1), ('d', 'e', 5 )]
my_dict[('SEX_ID', 'GENRE_NAME')]
            
        
for ccol in coupon_cols:
    c_variables = purchase_clist_ulist[ccol].unique()
    nested_joint_prob[ccol] = {}
    for ucol in user_cols:
        nested_joint_prob[ccol][ucol] = {}
        for variable in c_variables:
            ind = (purchase_clist_ulist[ccol] == variable)
            probs = purchase_clist_ulist[ucol][ind].value_counts()/sum(ind)
            nested_joint_prob[ccol][ucol][variable] = probs.to_dict()
            
        
for key in nested_joint_prob["GENRE_NAME"]["SEX_ID"].keys():
    for subkey, value in nested_joint_prob["GENRE_NAME"]["small_area_name"][key].items():
        print key, subkey, value


nested_joint_prob["GENRE_NAME"]["SEX_ID"]["健康・医療"]              





###############################################################################
###############################################################################
# functions to calculate joint probabilities
###############################################################################
###############################################################################
req_columns = ["GENRE_NAME", "SEX_ID"]
purchase_clist_ulist[req_columns]

joint_prob = {}
genres = purchase_clist_ulist.GENRE_NAME.unique()
genders = purchase_clist_ulist.SEX_ID.unique()
for genre in genres:
    ind = (purchase_clist_ulist.GENRE_NAME == genre)
    probs = purchase_clist_ulist.SEX_ID[ind].value_counts()/sum(ind)
    joint_prob[genre] = probs.to_dict()
    
for key in joint_prob.keys():
    for subkey, value in joint_prob[key].items():
        print key, subkey, value





        
###############################################################################
###############################################################################
# multiple joint probabilities
###############################################################################
###############################################################################
nested_joint_prob = {}
coupon_cols = ["GENRE_NAME", "SMALL_AREA_NAME"]
user_cols = ["SEX_ID", "small_area_name"]
for ccol in coupon_cols:
    c_variables = purchase_clist_ulist[ccol].unique()
    nested_joint_prob[ccol] = {}
    for ucol in user_cols:
        nested_joint_prob[ccol][ucol] = {}
        for variable in c_variables:
            ind = (purchase_clist_ulist[ccol] == variable)
            probs = purchase_clist_ulist[ucol][ind].value_counts()/sum(ind)
            nested_joint_prob[ccol][ucol][variable] = probs.to_dict()
            
        
for key in nested_joint_prob["GENRE_NAME"]["SEX_ID"].keys():
    for subkey, value in nested_joint_prob["GENRE_NAME"]["small_area_name"][key].items():
        print key, subkey, value


nested_joint_prob["GENRE_NAME"]["SEX_ID"]["健康・医療"]                

###############################################################################
# Creating discrete variable
###############################################################################
user_list.head()

variable = "age"
numbers = range(8)
variables = [variable + str(number) for number in numbers]

pd.cut(user_list.AGE, bins = [0,10,20,30,40,50,60,70,100], labels = variables)
user_list["age_group"] = pd.cut(user_list.AGE, bins = 
         [0,10,20,30,40,50,60,70,100], labels = variables)