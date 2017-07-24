# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 08:54:53 2017

@author: tiwarir
"""

import pandas as pd
from DBOperations import DBOperations as DBOps

class Get_Recommendation(object):
    def __init__(self, dbName, recommendation_type, user_id):
        self.db_name = dbName
        self.recommendation_type = recommendation_type
        self.user_id = user_id
        
    def start_process(self):
        recommendations = self.get_user_recommendation()
        return recommendations
        
        
    def get_user_recommendation(self):
        column_name  = 'user_id'
        
        if self.recommendation_type == 'collaborative':
            table_name = 'ResultantRatingMatrix'
            query = 'SELECT * FROM ' + table_name + ' WHERE ' + column_name  + '=\'' + self.user_id + '\''
            with DBOps(self.db_name) as simMatDbOps:
                df = pd.read_sql_query(query, simMatDbOps.getConnection())
                df = df.drop('user_id', axis = 1)
            df.sort_values(by = 0, ascending  = False, axis = 1, inplace = True)
            
        elif self.recommendation_type == 'content':            
            table_name = 'SimilarityMatrix'
            query = 'SELECT * FROM ' + table_name + ' WHERE ' + column_name  + '=\'' + self.user_id + '\''
            with DBOps(self.db_name) as simMatDbOps:
                df = pd.read_sql_query(query, simMatDbOps.getConnection())
                df = df.drop('user_id', axis = 1)
            df.sort_values(by = 0, ascending  = False, axis = 1, inplace = True)
            
        else:
            similarity_table = 'SimilarityMatrix'
            rating_table = 'ResultantRatingMatrix'
            with DBOps(self.db_name) as DbOps:
                query_similarity = 'SELECT * FROM ' + similarity_table + ' WHERE ' + column_name  + '=\'' + self.user_id + '\''
                df_similarity = pd.read_sql_query(query_similarity, DbOps.getConnection())
                query_rating = 'SELECT * FROM ' + rating_table + ' WHERE ' + column_name  + '=\'' + self.user_id + '\''
                df_rating = pd.read_sql_query(query_rating, DbOps.getConnection())
            df = df_similarity + df_rating
            
        return df.columns[:10].tolist()

###############################################################################        
if __name__ == '__main__' :
    getRecom = Get_Recommendation('ponpareDB', 'collaborative', 'c6f70990bb772ec55211b353389421b2')
    print "Collaborative Filtering: ", getRecom.start_process()
#    getRecom = Get_Recommendation('ponpareDB', 'content', '684ca5035f10f1e156803a3fd245b5c4')
#    print "Content Filtering: ", getRecom.start_process()
        
        