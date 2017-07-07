# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:17:30 2017

@author: tiwarir
"""

###############################################################################
# change to the appropriate working directory
###############################################################################
from __future__ import division
import os
import pandas as pd


###############################################################################
# chagne the working directory
###############################################################################
os.getcwd()
new_dir = os.path.join("C:\\","USERS", "tiwarir", "Documents", "ponpare", "hybrid_filtering")
os.chdir(new_dir)
os.listdir(".")
###############################################################################
# setting display options
###############################################################################
pd.options.display.max_columns = 100
pd.set_option('expand_frame_repr', False)

###############################################################################
# importing the required modules
###############################################################################
import collaborative_filtering_functions as colf
import content_filtering_functions as conf
import data_loading as dl
import data_preprocessing as dpre
import data_postprocessing as dpost
import subset_data as sd
import create_training_test_data as cttd
import popular_item_recommendations as popi

reload(colf)
reload(conf)
reload(dl)
reload(sd)
reload(cttd)
reload(dpost)
reload(dpre)


coupon_visit = dl.load_coupon_visit_data()
coupon_visit_subset = sd.create_data_subset(n_users = 1000, min_purchase = 1, max_purchase = 5, seed_value = 10)
coupon_clust_visit = dpre.substitute_coupon_id_with_cluster_id(coupon_visit_subset)
train, test = cttd.create_train_test_set(coupon_clust_visit, train_frac = 0.7, seed_value = 10)

# collaborative filtering
colf_users_recommendation = colf.get_collaborative_filtering_recommendation(train, test)

# content filtering
#conf_recommendation_dict = conf.get_recommendation_for_test_data(test) 

# popular item recommendation
#pop_users_recommendation = popi.create_recommendation_based_on_popular_item(train, test)

# result analysis
view_purchase_dict = dpost.item_viewed_purchased(train, test)
per_accuracy =  dpost.calculate_percentage_accuracy(colf_users_recommendation, view_purchase_dict)
print per_accuracy


