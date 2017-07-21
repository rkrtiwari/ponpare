# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:32:43 2017

@author: tiwarir
"""
from __future__ import division
import numpy as np
import pandas as pd
from DBOperations import DBOperations as DBOps


class Content_Filtering(object):
    def __init__(self, dbName):
        self.db_name = dbName

    def start_process(self):
        print 'Content Filter processing started...'
        self.get_user_content_vector()
        self.get_coupon_content_vector()
        self.store_recommendations_in_db()
        print ' Content Filter processing finished...'

# user content vector creation         
    def load_user_data(self):
        print 'Loading User Data from DB.. Started...'
        with DBOps(self.db_name) as dbOps:
            ui_df = dbOps.get_all_records_as_df(table_name = 'UserInformation')
        print ' Loading User Data from DB.. Finished...'
        return ui_df
    
    def get_user_content_vector(self, age_cat, sex_cat):
        a_v = np.zeros(6)
        ind = int(age_cat[3])
        a_v[ind] = 1
    
        if sex_cat == 'm':
            s_v = np.array([1,0])
        else:
            s_v = np.array([0,1])
        u_v = np.concatenate([s_v,a_v])
        return u_v

    def create_user_vector_dict(self, u_df):    
       
        user_content_vector_dict = {}
        n_users, n_features = u_df.shape
        
        for i in range(n_users):
            user_id = u_df.USER_ID_hash.iat[i]
            gender = u_df.SEX_ID.iat[i]
            age = u_df.AGE_CATEGORY.iat[i]
            user_vector = self.get_user_content_vector(age, gender)
            user_content_vector_dict[user_id] = user_vector
        return user_content_vector_dict

# coupon content vector
    def load_coupon_cluster_info_data(self):
        print 'Loading Coupon Data from DB.. Started...'
        with DBOps(self.db_name) as dbOps:
            cci_df = dbOps.get_all_records_as_df(table_name = 'CouponClustInfo')
        print ' Loading Coupon Cluster Info from DB.. Finished...'
        return cci_df

    def load_user_purchased_coupon_data(self):
        print 'Loading User Coupon Activity Data from DB.. Started...'
        with DBOps(self.db_name) as dbOps:
            up_df = dbOps.conditional_selection('UserCouponActivityInfo', 'PURCHASE_FLG', 1)
        print ' Loading User Coupon Activity Data from DB.. Finished...'
        return up_df


    def get_conditional_probability(self, up_df, ui_df, cci_df):
        
        print "calculating conditional probability"
        
        int_df = up_df.merge(ui_df, on = 'USER_ID_hash')
        coupon_purchase_data = int_df.merge(cci_df, on = 'CLUSTER_ID')
        coupon_purchase_data = coupon_purchase_data.dropna(axis = 0, how = 'any')
        
        c_features = ["GENRE_NAME", "PRICE_RATE_CATEGORY", "PRICE_CATEGORY"]
        u_features = ["AGE_CATEGORY", "SEX_ID"]
        
        coupon_cond_prob = pd.DataFrame(columns = ('coupon_feature', 'user_feature', 
                                               'coupon_feature_value', 'user_feature_value',
                                               'cond_prob'))
        
        i = 0
        for c_feature in c_features:
            c_feature_values = coupon_purchase_data[c_feature].unique()
            c_value_count =  coupon_purchase_data[c_feature].value_counts()
            c_total = sum(c_value_count)
            for c_feature_value in c_feature_values:
                c_prob =  c_value_count.loc[c_feature_value]/c_total
                for u_feature in u_features:
                    u_feature_values = coupon_purchase_data[u_feature].unique()
                    u_value_count =  coupon_purchase_data[u_feature].value_counts()
                    u_total = sum(u_value_count)
                    
                    ind = coupon_purchase_data[c_feature] == c_feature_value
                    u_value_count_cond =  coupon_purchase_data[ind][u_feature].value_counts()
                    u_total_cond = sum(u_value_count_cond)
                    
                    for u_feature_value in u_feature_values:
                        u_prob =  u_value_count.loc[u_feature_value]/u_total
                        if u_feature_value not in u_value_count_cond:
                            coupon_cond_prob.loc[i] = [c_feature, u_feature, c_feature_value,
                                        u_feature_value, 0.00000001]
                            i += 1
                            continue
                        u_prob_cond = u_value_count_cond.loc[u_feature_value]/u_total_cond
                        post_prob = c_prob*u_prob_cond/u_prob
                        coupon_cond_prob.loc[i] = [c_feature, u_feature, c_feature_value,
                                        u_feature_value, post_prob]
                        i += 1
        coupon_cond_prob.sort_values(by = ['coupon_feature', 'user_feature', 
                    'coupon_feature_value', 'user_feature_value'],  inplace = True)
        return coupon_cond_prob



    def create_coupon_content_vector_dict(self, cp_df, cci_df):
        print "creating coupon vector dictionary"    
        
        coupon_content_vector_dict = {}
        nrow, _ = cci_df.shape
        for i in range(nrow):
            key  = cci_df.CLUSTER_ID.iat[i]
            genre = cci_df.GENRE_NAME.iat[i]
            price = cci_df.PRICE_CATEGORY.iat[i]
            discount = cci_df.PRICE_RATE_CATEGORY.iat[i]
            ind = cp_df.coupon_feature_value == genre
            g_v = cp_df.cond_prob.loc[ind].values
            ind = cp_df.coupon_feature_value == price
            p_v = cp_df.cond_prob.loc[ind].values
            ind = cp_df.coupon_feature_value == discount
            d_v = cp_df.cond_prob.loc[ind].values
            i_v = np.dot(g_v,p_v)
            v = np.dot(i_v,d_v)
            coupon_content_vector_dict[key] = v
        return coupon_content_vector_dict    

    def create_similarity_matrix(self, user_list, user_content_vector_dict, coupon_content_vector_dict):
        coupons = coupon_content_vector_dict.keys()
        n_row = len(user_list)
        n_col = len(coupons)
        data = np.zeros((n_row, n_col))
        similarity_matrix = pd.DataFrame(data, index = user_list, columns = coupons)
        for user in user_list:
            for coupon in coupons:
                user_vector = user_content_vector_dict[user]
                coupon_vec = coupon_content_vector_dict[coupon]
                distance = np.dot(user_vector, coupon_vec)
                similarity_matrix.loc[user, coupon] = distance
        return similarity_matrix

    def storeSimilarityMatrixInDB(self, similarity_matrix):
        similarity_matrix['index1'] = similarity_matrix.index
        print 'Storing Similarity Matrix to DB:', self.db_name,'Started...'
        with DBOps(self.db_name) as simMatDbOps:
            similarity_matrix.to_sql('SimilarityMatrix', simMatDbOps.getConnection(), index = False, if_exists = 'replace')
        print ' Storing Similarity Matrix to DB:', self.db_name,'Finished...'
        
    def get_user_recommendation(self, user_id):
        table_name = 'SimilarityMatrix'
        column_name = 'index1'
#        query = 'SELECT * FROM ' + table_name + ' WHERE ' + column_name  + '=\"' + user_id   + '\"'
        query = 'SELECT * FROM ' + table_name + ' WHERE ' + column_name  + '=\'' + user_id + '\''
        with DBOps(self.db_name) as simMatDbOps:
            df = pd.read_sql_query(query, simMatDbOps.getConnection())
        df.sort_values(by = 0, ascending  = False, axis = 1, inplace = True)
        return df.iloc[:,:10]





if __name__ == '__main__':
    conF = Content_Filtering('ponpareDB') 
    ui_df = conF.load_user_data()
    cci_df = conF.load_coupon_cluster_info_data()
    up_df = conF.load_user_purchased_coupon_data()
    user_content_vector_dict = conF.create_user_vector_dict(ui_df)
    cp_df = conF.get_conditional_probability(up_df, ui_df, cci_df)
    coupon_content_vector_dict = conF.create_coupon_content_vector_dict(cp_df, cci_df)
    all_users = ui_df.USER_ID_hash.tolist()
    user_list = np.random.choice(all_users, 20)
    similarity_matrix = conF.create_similarity_matrix(user_list, user_content_vector_dict, coupon_content_vector_dict)
    conF.storeSimilarityMatrixInDB(similarity_matrix)
    user_id = user_list[0]
    df = conF.get_user_recommendation(user_id)
    
    
    
#    uca_df = conF.load_user_coupon_activity_data()
    


