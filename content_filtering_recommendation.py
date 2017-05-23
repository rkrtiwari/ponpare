# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:16:53 2017

@author: tiwarir
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

###############################################################################
# make a recommendation. send the sorted list for recommendation
###############################################################################
#1. create a ranking for a given user
coupon_content_vector_dict
user_content_vector_dict
purchase_area

###############################################################################
# view few items in the dictionary to have some idea
###############################################################################
dict(user_content_vector_dict.items()[0:2])
dict(coupon_content_vector_dict.items()[0:2])
dict(purchase_area.items()[0:2])



###############################################################################
# 1. Choose a random user
###############################################################################
def get_a_user(user_list, user_content_vector_dict, test):
    ind = pd.isnull(user_list.PREF_NAME)
    users = user_list.USER_ID_hash[~ind] 
    test_users = [user for user in users if user in test.USER_ID_hash.values]
#    users = user_content_vector_dict.keys()
    n_users = len(test_users)
    ind = np.random.choice(n_users, size = 1)
    user_id = test_users[ind]
    user_vector = user_content_vector_dict[user_id]
    return user_id, user_vector




###############################################################################
# module to get a random user's coupon ranking
###############################################################################
def create_product_ranking(user_content, coupon_content_vector_dict, purchased_area):
    user_vector = user_content[0]
    user_pref = user_content[1]
    ken_area = purchased_area[user_pref]
    areas = ken_area.keys()
    
    coupon_ranking = pd.DataFrame(columns = ('coupon_id', 'ken_area',
                                         'score'))
    i = 0
    for coupon_id in coupon_content_vector_dict.keys():
        coupon_vec = coupon_content_vector_dict[coupon_id][0]
        coupon_ken = coupon_content_vector_dict[coupon_id][1]
        coupon_val = coupon_content_vector_dict[coupon_id][2]
        if coupon_ken in areas:
            score = np.dot(user_vector, coupon_vec)
            score *= ken_area[coupon_ken]/100
            coupon_ranking.loc[i] = [coupon_id, coupon_ken, score]
            i +=1
    
    coupon_ranking.sort_values(by = 'score', axis = 0, ascending = False,
                           inplace = True)
    return coupon_ranking      
            
user_id, user_content = get_a_user(user_list, user_content_vector_dict, test)            
product_rank = create_product_ranking(user_content, coupon_content_vector_dict, purchased_area)
coupons = product_rank.coupon_id.values
ind = coupon_list_train.COUPON_ID_hash == coupons
coupon_list_train[['COUPON_ID_hash', 'GENRE_NAME', 'price_cat','price_rate_cat']].loc[ind]


###############################################################################
# code for validation
###############################################################################
#1. recommended items

def get_recommendation(user_id, user_content):
    product_rank = create_product_ranking(user_content, coupon_content_vector_dict, purchased_area)
    product_rank = product_rank.drop_duplicates(subset = 'score', keep = 'first')
    coupons = product_rank.coupon_id.iloc[:10]
    recommendation = []
    for coupon in coupons:
        ind = coupon_list_train.COUPON_ID_hash == coupon
        coupon_feat = (coupon_list_train.price_rate_cat.loc[ind].values[0], \
          coupon_list_train.price_cat.loc[ind].values[0],\
          coupon_list_train.GENRE_NAME.loc[ind].values[0])
        recommendation.append(coupon_feat)
    return recommendation


#2. item bought during the test period        
def get_items_bought(user_id):
    purchased = []
    ind = test.USER_ID_hash == user_id
    actual_purchase = test[['GENRE_NAME', 'price_cat', 'price_rate_cat']].loc[ind]
    for i in range(len(actual_purchase)):
        pur_coup_feat = (actual_purchase.price_rate_cat.values[i], \
                     actual_purchase.price_cat.values[i],\
                     actual_purchase.GENRE_NAME.values[i])
        purchased.append(pur_coup_feat)
    return purchased 


        
#3. function to check accuracy
def check_accuracy(recommendation, purchased):
    s1 = set(recommendation)
    s2 = set(purchased)
    s3 = s2.intersection(s1)
    if (len(s2) > 0):
        return len(s2), len(s3)*100/len(s2)
    else:
        return len(s2), 'no purchase'

# 4 get accuracy for 100 users
def get_accuracy_multiple_users(n=5):
    accuracy = []
    for i in xrange(n):
        print i
        user_id, user_content = get_a_user(user_list, user_content_vector_dict, test)
        recommendation = get_recommendation(user_id, user_content)
        purchased =  get_items_bought(user_id)
        accuracy.append(check_accuracy(recommendation, purchased))
    return accuracy
        
accuracy =  get_accuracy_multiple_users(n=5) 
accuracy_string = str(accuracy)
f =  open('accuracy.txt', 'wb ') 
f.write(accuracy_string) 
f.close() 

pickle.dump(accuracy, open('accuracy.pkl', 'wb '))
accuracy2 = pickle.load(open('accuracy.pkl','rb'))
accuracy2 

df = pd.DataFrame(columns=['purchase_count', 'correct_prediction'])

i = 0
for items in accuracy2:
    no, percent = items
    df.loc[i] =  [no, int(no*percent/100)]
    i += 1

df.head()      
    
# zero correct
zero_right = df.loc[df.correct_prediction ==0,]
one_right = df.loc[df.correct_prediction ==1,]
two_right = df.loc[df.correct_prediction ==2,]
three_right = df.loc[df.correct_prediction ==3,] 

#1. zero match
title = "total cases of zero match:", len(zero_right)
plt.hist(zero_right.purchase_count.values, align = 'left', bins = [1,2,3,4,5,6],
         rwidth = 0.9, normed = False)
plt.title(title)
plt.xlabel("user purchase count")
plt.show()

#2. one match
title = "total cases of one match:", len(one_right)
plt.hist(one_right.purchase_count.values, align = 'left', bins = [1,2,3,4,5,6],
         rwidth = 0.9, normed = False)
plt.title(title)
plt.xlabel("user purchase count")
plt.show()

#2. two match
title = "total cases of two match:", len(two_right)
plt.hist(two_right.purchase_count.values, align = 'left', bins = [1,2,3,4,5,6],
         rwidth = 0.9, normed = False)
plt.title(title)
plt.xlabel("user purchase count")
plt.yticks([1,2])
plt.show()

#3. three match
title = "total cases of three match:", len(three_right)
plt.hist(three_right.purchase_count.values, align = 'left', bins = [1,2,3,4,5,6],
         rwidth = 0.9, normed = False)
plt.title(title)
plt.xlabel("user purchase count")
plt.yticks([1])
plt.show()


    
    
    
    
    
    
    

  

      
   






