# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:57:03 2017

@author: tiwarir
"""

from __future__ import division
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from DBOperations import DBOperations as DBOps

class Collaborative_Filtering(object):
    def __init__(self, dbName):
        self.db_name = dbName

    def start_process(self):
        print 'CF processing started...'
        uca_df = self.load_user_coupon_activity_data()
        intermediate_rating_matrix = self.create_intermediate_rating_matrix(uca_df)
        resultant_rating_matrix = self.create_resultant_rating_matrix(intermediate_rating_matrix)
        self.store_resultant_matrix_in_db(resultant_rating_matrix)
        print ' CF processing finished...'
        return


    def create_intermediate_rating_matrix(self, uca_df):
        print 'Creating Intermediate Rating Matrix.. Started...'
        train1 = uca_df.copy()
        train1['RATING'] = 0.00

        n_purchased = np.sum(train1.PURCHASE_FLG == 1)
        n_visited = np.sum(train1.PURCHASE_FLG == 0)

        visited_rating = float(n_purchased)/float(n_visited)
        ind_visit  = train1.PURCHASE_FLG == 0
        train1.loc[ind_visit, 'RATING'] = visited_rating

        ind_purchase = train1.PURCHASE_FLG == 1
        train1.loc[ind_purchase, 'RATING'] = 1

        train2 = train1.groupby(by = ['USER_ID_hash', 'CLUSTER_ID'], as_index = False).sum()
        rating_matrix = train2.pivot(index = 'USER_ID_hash', columns = 'CLUSTER_ID', values = 'RATING')
        rating_matrix = rating_matrix.fillna(value = 0)

        print ' Creating Intermediate Rating Matrix.. Finished...'
        return rating_matrix

    def create_resultant_rating_matrix(self, intermediate_rating_matrix, n_comp = 5):
        print 'Creating Resultant Rating Matrix.. Started...'
        R = intermediate_rating_matrix.values
        model = NMF(n_components = n_comp, init = 'random', random_state = 0)
        W = model.fit_transform(R)
        H = model.components_
        R_full = np.dot(W,H)
        final_rating_matrix = pd.DataFrame(R_full, index = intermediate_rating_matrix.index,
                                   columns = intermediate_rating_matrix.columns)
        print ' Creating Resultant Rating Matrix.. Finished...'
        return final_rating_matrix

    def load_user_coupon_activity_data(self):
        print 'Loading User Coupon Activity Data from DB.. Started...'
        with DBOps(self.db_name) as dbOps:
            uca_df = dbOps.get_all_records_as_df(table_name = 'UserCouponActivityInfo')
            print ' Loading User Coupon Activity Data from DB.. Finished...'
            return uca_df

    def store_resultant_matrix_in_db(self, matrix_df):
        print 'Storing Resultant Rating Matrix to DB.. Started...'
        with DBOps(self.db_name) as matrix_DbOps:
            matrix_df.to_sql('ResultantRatingMatrix', matrix_DbOps.getConnection(), index = False, if_exists = 'replace')
        print ' Storing Resultant Rating Matrix to DB.. Finished...'



if __name__ == '__main__':
    colF = Collaborative_Filtering('ponpareDB')
    colF.start_process()
#    uca_df = colF.load_user_coupon_activity_data()
#    intermediate_rating_matrix  = colF.create_intermediate_rating_matrix(uca_df)
#    resultant_rating_matrix = colF.create_resultant_rating_matrix(intermediate_rating_matrix)
#    colF.store_resultant_matrix_in_db(resultant_rating_matrix)



