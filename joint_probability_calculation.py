# -*- coding: utf-8 -*-
"""
Created on Thu May 04 16:14:44 2017

@author: ravitiwari
"""

###############################################################################
###############################################################################
# functions to calculate probabilities: part 1
###############################################################################
###############################################################################

prob = {}

user_list.columns

i_columns = [1,4]
user_list.columns[i_columns]

## probability generation
for i in i_columns:
    gen_prob = user_list.iloc[:,i].value_counts()/len(user_list)
    prob[user_list.columns[i]] = gen_prob.to_dict()

## printing probabilities
for key in prob.keys():
    for subkey, value in prob[key].items():
        print key, subkey, value
        

    
###############################################################################
###############################################################################
# functions to calculate continuous probabilities 
###############################################################################
###############################################################################
user_list.columns
plt.hist(user_list.AGE, bins = 20)
mu, std = norm.fit(user_list.AGE)

plt.hist(user_list.AGE, bins=25, normed=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.show()


coupon_list_train.columns
coupon_list_train.CATALOG_PRICE

alpha, beta, loc, scale = beta.fit(coupon_list_train.CATALOG_PRICE)
###############################################################################
################################################################################
###############################################################################
# function to calculate probabilities for coupons based on purchase data
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

###############################################################################
###############################################################################
###############################################################################
# conditional joint probability  
###############################################################################
###############################################################################
###############################################################################

coupon_cols = ["GENRE_NAME", "SMALL_AREA_NAME"]
user_cols = ["SEX_ID", "small_area_name"]

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