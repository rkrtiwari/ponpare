# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:03:14 2017

@author: tiwarir
"""
test
test_0

# checking if the training and the test data are the same 
#1. if the shape are the same for the training and test data

# if all the elements in the training are the same
if (train.size == np.sum(np.sum(train == train_0))):
    print "Training data is the same"
else:
    print "Training data are NOT the same"
    
# if all the elements are in the test data are the same
if (test.size == np.sum(np.sum(test == test_0))):
    print "Test data is the same"
else:
    print "Test data are NOT the same"

# checking if all the elements of the rating matrix are the same
rating_matrix = colf.create_rating_matrix(train)
if (rating_matrix.size == np.sum(np.sum(rating_matrix == rating_matrix_0))):
    print "rating matrices are the same"
else:
    print "rating matrices are NOT the same"

# checking if all the elements of the final rating matrics are the same
R_final = colf.create_final_rating_matrix(rating_matrix, n_comp = 5)
if (R_final.size == np.sum(np.sum(R_final == R_final_0))):
    print "Final Rating matrices are the same"
else:
    print "Final Rating matrices are NOT the same"

# testing recommendation for test users one by one
test_users = test.USER_ID_hash.unique().tolist()
train_users = train.USER_ID_hash.unique().tolist()
test_0_users = test_0.USER_ID_hash.unique().tolist()
train_0_users = train_0.USER_ID_hash.unique().tolist()

# are the train users the same
if (train_users == train_0_users):
    print "train users are the same"
else:
    print "train users are NOT the same"

# are the test users the same
if (test_users == test_0_users):
    print "test users are the same in both the cases"
else:
    print "test users are NOT the same in both the cases"

# test users in the training data
test_users_in_training = [user for user in test_users if user in train_users]
test_users_not_in_training = [user for user in test_users if user not in train_users]

# checking recommendation for individual users
for user in test_users:
    recommendation = colf.get_recommendation_for_a_user(R_final, user)
    print "USER_ID:", user
    print recommendation
    print "\n\n"
    recommendation_0 = cf.get_recommendation_for_a_user(R_final, user)
    print recommendation_0
    print "____________________________________________________________________"
    if recommendation == recommendation_0:
        print "recommendations are the same"
    else:
        print "recommendations DIFFER"
    print "____________________________________________________________________"

recommendation_dicts = colf.create_recommendation_for_test_data(train, test, n_comp = 5)
for user in test_users:
    recommendation = colf.get_recommendation_for_a_user(R_final, user)
    print "USER_ID:", user
    print recommendation
    print "\n\n"
    recommendation_dict = recommendation_dicts[user]
    print recommendation_dict
    print "____________________________________________________________________"
    if recommendation == recommendation_dict:
        print "recommendations are the same"
    else:
        print "recommendations DIFFER"
    print "____________________________________________________________________"


recommendation_dicts = cf.create_recommendation_for_test_data(train, test, n_comp = 5)
for user in test_users:
    recommendation = cf.get_recommendation_for_a_user(R_final_0, user)
    print "USER_ID:", user
    print recommendation
    print "\n\n"
    recommendation_dict = recommendation_dicts[user]
    print recommendation_dict
    print "____________________________________________________________________"
    if recommendation == recommendation_dict:
        print "recommendations are the same"
    else:
        print "recommendations DIFFER"
    print "____________________________________________________________________"



colf_users_recommendation = colf.create_recommendation_for_test_data(train, test, n_comp = 5)
cf_recommendations_dict_0 = cf.create_recommendation_for_test_users(test_0, rating_matrix_0, R_final_0)


for user in test_users:
    recommendation = colf_users_recommendation[user]
    recommendation_0 = cf_recommendations_dict_0[user]
    print "user_id:", user
    for rec, rec_0 in zip(recommendation, recommendation_0):
        print rec, rec_0
    print "____________________________________________________________________"   
    if (recommendation == recommendation_0):
        print "both the method have the same recommendation"
    else:
        print "recommendations DO NOT match"
    print "____________________________________________________________________"
        
    


# checking if the elements in the view purchase dictionary are the same
# test data
view_purchase_dict == view_purchase_dict_0
colf_users_recommendation == cf_recommendations_dict_0
test_use
