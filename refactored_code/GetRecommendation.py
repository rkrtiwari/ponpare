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
        if self.recommendation_type == 'collaborative':
            recommendation = self.get_collaborative_recommendation()
            recommendation = recommendation.T
        if self.recommendation_type == 'content':
            recommendation = self.get_content_filtering_recommendation()
            recommendation = recommendation.T
        if self.recommendation_type == 'hybrid':
            recommendation = self.get_hybrid_filtering_recommendation()
        recommendation.columns = ['score']
        return recommendation.index[:10].tolist()
#        return recommendation.iloc[:10,]
    

    
    def get_collaborative_recommendation(self):
        column_name  = 'user_id'
        table_name = 'ResultantRatingMatrix'
        query = 'SELECT * FROM ' + table_name + ' WHERE ' + column_name  + '=\'' + self.user_id + '\''
        with DBOps(self.db_name) as simMatDbOps:
            df_col = pd.read_sql_query(query, simMatDbOps.getConnection())
            df_col = df_col.drop('user_id', axis = 1)
        df_col.sort_values(by = 0, ascending  = False, axis = 1, inplace = True)
        return df_col
    
    def get_content_filtering_recommendation(self):
        column_name  = 'user_id'
        table_name = 'SimilarityMatrix'
        query = 'SELECT * FROM ' + table_name + ' WHERE ' + column_name  + '=\'' + self.user_id + '\''
        with DBOps(self.db_name) as simMatDbOps:
            df_con = pd.read_sql_query(query, simMatDbOps.getConnection())
            df_con = df_con.drop('user_id', axis = 1)
        df_con.sort_values(by = 0, ascending  = False, axis = 1, inplace = True)
        return df_con
        
    def get_hybrid_filtering_recommendation(self):
        col_recom = self.get_collaborative_recommendation()
        col_recom = col_recom.T
        col_min = col_recom.min()
        col_max = col_recom.max()
        col_recom = (col_recom - col_min)/(col_max - col_min)
        
        con_recom = self.get_content_filtering_recommendation()
        con_recom = con_recom.T
        con_min = con_recom.min()
        con_max = con_recom.max()
        con_recom = (con_recom - con_min)/(con_max - con_min)
        
        recom = col_recom + con_recom
        recom.sort_values(by = 0, ascending = False, inplace = True)
        return recom
        

###############################################################################        
if __name__ == '__main__' :    
    dbname = 'ponpareDB'
    user_id = '002383753c1e5d6305c8aff6f89e26d6'
    
    getRecom = Get_Recommendation(dbname, 'collaborative', user_id)
    df_col = getRecom.start_process()
    
    getRecom = Get_Recommendation(dbname, 'content', user_id)
    df_con = getRecom.start_process()
    
    getRecom = Get_Recommendation(dbname, 'hybrid', user_id)
    df_hybrid = getRecom.start_process()



    
        
        