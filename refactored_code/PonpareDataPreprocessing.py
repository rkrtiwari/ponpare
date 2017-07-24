# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:57:03 2017

@author: tiwarir
"""

import pandas as pd
import numpy as np
import os
from DBOperations import DBOperations as DBOps

""" This class is used to preprocess the ponpare data """
class PonpareDataPreprocessing(object):
    def __init__(self, dataPath, dbName):
        self.dataPath = dataPath
        self.dbName = dbName


    def startProcess(self):
        print 'Beginning Ponpare DataPreprocessing...'
        print 'Picking the required data from the path: ', self.dataPath
        
        # process user information and load it into database
        user_list = self.loadUserListData()
        user_list_age_cat = self.generateAgeCategoryForUsers(user_list)
        self.storeUserInfoInDB(user_list_age_cat)
        
        # process coupon information and store it into database
        coupon_list = self.loadCouponListTrainData()
        coupon_list_price_cat = self.generatePriceCategoryForCoupons(coupon_list)
        coupon_list_all_cat  = self.generateDiscoutCategoryForCoupons(coupon_list_price_cat)
        self.storeCouponInfoInDB(coupon_list_all_cat)
        
        # find the cluster id of all the coupons and save it into database 
        coupon_id_2_cluster_id = self.createCouponId2ClusterId(coupon_list_all_cat)
        self.storeCouponId2ClusterIdInDB(coupon_id_2_cluster_id) 
        
        # find the cluster information and save it into database
        coupon_clust_def = self.createCouponClusterDef(coupon_list_all_cat)
        self.storeCouponClustDefInDB(coupon_clust_def)
        
        # process coupon visit data and save it into database
        coupon_visit = self.loadCouponVisitTrainData()
        pruned_coupon_visit = self.pruneCouponVisitTrain(coupon_visit, coupon_list)
        pruned_coupon_cluster_visit = self.addClusterIdPrunedCouponVisit(pruned_coupon_visit, coupon_id_2_cluster_id)
        self.storeUserCouponActivityInfo(pruned_coupon_cluster_visit)


    def loadUserListData(self):
        print 'Loading User List Data. Started...'
        userListCSVFilePath = os.path.join(self.dataPath, 'user_list.csv')
        fields = ['USER_ID_hash', 'AGE', 'SEX_ID']
        user_list = pd.read_csv(userListCSVFilePath, usecols = fields)
        print ' Loading User List Data. Finished...'
        return user_list
    
    def generateAgeCategoryForUsers(self, user_list):
        ageCategories = [0, 20, 30, 40, 50, 60, 100]
        sufs = np.arange(len(ageCategories) - 1)
        ageLabels = map(lambda x: 'age'+str(x), sufs)
        user_list_age_cat = user_list.copy()
        user_list_age_cat['AGE_CATEGORY'] = pd.cut(user_list_age_cat['AGE'], bins = ageCategories, labels = ageLabels)
        return user_list_age_cat
    
    def storeUserInfoInDB(self, user_list_age_cat):
        print 'Storing the User Information to DB:', self.dbName,'Started...'
        with DBOps(self.dbName) as userInfoDbOps:
            user_list_age_cat.to_sql('UserInformation', userInfoDbOps.getConnection(), index = False, if_exists = 'replace')
        print ' Storing the User Information to DB:', self.dbName,'Finished...'


    def loadCouponListTrainData(self):
        print 'Loading Coupon List Train Data. Started...'
        couponListCSVFilePath = os.path.join(self.dataPath, 'coupon_list_train.csv')
        fields  = ['COUPON_ID_hash', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE']
        coupon_list_train = pd.read_csv(couponListCSVFilePath, usecols = fields)
        print ' Loading Coupon List Train Data. Finished...'
        return coupon_list_train

    def generatePriceCategoryForCoupons(self, coupon_list):
        priceCategories = [0, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 1000000]
        sufs = np.arange(len(priceCategories) - 1)
        priceLabels = map(lambda x: 'catalog_price'+str(x), sufs)
        coupon_list_price_cat  = coupon_list.copy()
        coupon_list_price_cat['PRICE_CATEGORY'] = pd.cut(coupon_list_price_cat['CATALOG_PRICE'], bins = priceCategories, labels = priceLabels)
        return coupon_list_price_cat

    def generateDiscoutCategoryForCoupons(self, coupon_list_price_cat):
        discountCategories = [-1, 25, 50, 60, 70, 80, 90, 100]
        sufs = np.arange(len(discountCategories) - 1)
        discountLabels = map(lambda x: 'price_rate'+str(x), sufs)
        coupon_list_all_cat = coupon_list_price_cat.copy() 
        coupon_list_all_cat['PRICE_RATE_CATEGORY'] = pd.cut(coupon_list_all_cat['PRICE_RATE'], bins = discountCategories, labels = discountLabels)
        return coupon_list_all_cat

    def storeCouponInfoInDB(self, coupon_list_all_cat):
        print 'Storing the Coupon Information to DB:', self.dbName,'Started...'
        with DBOps(self.dbName) as couponInfoDbOps:
            coupon_list_all_cat.to_sql('CouponInformation', couponInfoDbOps.getConnection(), index = False, if_exists = 'replace')
        print ' Storing the Coupon Information to DB:', self.dbName,'Finished...'

    def createCouponId2ClusterId(self, coupon_list):
        featureSet = ['GENRE_NAME', 'PRICE_CATEGORY', 'PRICE_RATE_CATEGORY']
        coupon_list = coupon_list.sort_values(by = ['COUPON_ID_hash'])

        def createClusterId(x):
            x['CLUSTER_ID'] = x.COUPON_ID_hash.values[0]
            return x

        coupon_id_cluster = coupon_list.groupby(featureSet).apply(lambda x: createClusterId(x))
        columns_to_keep = ['COUPON_ID_hash', 'CLUSTER_ID']
        return coupon_id_cluster[columns_to_keep]

    def storeCouponId2ClusterIdInDB(self, coupon_id_2_cluster_id):
        print 'Storing the cluster id of coupons Info to DB:', self.dbName,'Started...'
        with DBOps(self.dbName) as clusterIDofCoupons:
            coupon_id_2_cluster_id.to_sql('clusterIdofCoupons', clusterIDofCoupons.getConnection(), index = False, if_exists = 'replace')
        print ' Storing the cluster id of coupons Info to DB:', self.dbName,'Finished...'
        
    def createCouponClusterDef(self, coupon_list):
        coupon_list = coupon_list.sort_values(by = 'COUPON_ID_hash')
        cluster_info_df = coupon_list.drop_duplicates(subset = ['GENRE_NAME','PRICE_CATEGORY', 'PRICE_RATE_CATEGORY'], keep = 'first')
        n, _ = cluster_info_df.shape
        fields = ['CLUSTER_ID','GENRE_NAME', 'PRICE_CATEGORY', 'PRICE_RATE_CATEGORY']
        coupon_clust_def = pd.DataFrame(columns = fields) 
        for  i in range(n):
            coupon_id =  cluster_info_df.COUPON_ID_hash.iloc[i]
            genre =  cluster_info_df.GENRE_NAME.iloc[i]
            discount =  cluster_info_df.PRICE_RATE_CATEGORY.iloc[i]
            price =  cluster_info_df.PRICE_CATEGORY.iloc[i]
            coupon_clust_def.loc[i] = [coupon_id, genre, price, discount]
        return coupon_clust_def

    def storeCouponClustDefInDB(self, coupon_clust_def):
        print 'Storing the Coupon Cluster Information to DB:', self.dbName,'Started...'
        with DBOps(self.dbName) as userInfoDbOps:
            coupon_clust_def.to_sql('CouponClustInfo', userInfoDbOps.getConnection(), index = False, if_exists = 'replace')
        print ' Storing the Coupon Cluster Information to DB:', self.dbName,'Finished...'        

    def loadCouponVisitTrainData(self):
        print 'Loading Coupon Visit Train Data. Started...'        
        couponVisitCSVFilePath = os.path.join(self.dataPath, 'coupon_visit_train.csv')
        fields  = ['USER_ID_hash', 'VIEW_COUPON_ID_hash', 'PURCHASE_FLG']
        coupon_vist_train = pd.read_csv(couponVisitCSVFilePath, usecols = fields)
        print ' Loading Coupon Visit Train Data. Finished...'
        return coupon_vist_train
    
    def pruneCouponVisitTrain(self, coupon_visit, coupon_list):
        coupons = coupon_list.COUPON_ID_hash.unique().tolist()
        ind = coupon_visit.VIEW_COUPON_ID_hash.isin(coupons)
        pruned_coupon_visit = coupon_visit.loc[ind]
        return pruned_coupon_visit

    def addClusterIdPrunedCouponVisit(self, pruned_coupon_visit, coupon_id_2_cluster_id):
        
        coupon_id_2_cluster_id_dict = coupon_id_2_cluster_id.set_index('COUPON_ID_hash').T.to_dict('list')
        
        def replaceWithClusterId(x):
            x['CLUSTER_ID'] = coupon_id_2_cluster_id_dict[x.VIEW_COUPON_ID_hash.iat[0]][0]
            return x

        pruned_coupon_cluster_visit = pruned_coupon_visit.groupby('VIEW_COUPON_ID_hash').apply(lambda x: replaceWithClusterId(x))
        pruned_coupon_cluster_visit = pruned_coupon_cluster_visit.rename(columns={"VIEW_COUPON_ID_hash": "COUPON_ID_old", "CLUSTER_ID": "COUPON_ID"})
        return pruned_coupon_cluster_visit        
    
    def storeUserCouponActivityInfo(self, pruned_coupon_cluster_visit):
        print 'Storing the User Coupon Activity Info to DB:', self.dbName,'Started...'
        with DBOps(self.dbName) as userCouponActivityDbOps:
            pruned_coupon_cluster_visit.to_sql('UserCouponActivityInfo', userCouponActivityDbOps.getConnection(), index = False, if_exists = 'replace')
        print ' Storing the User Coupon Activity Info to DB:', self.dbName,'Finished...'



################################################################################
if __name__ == '__main__':
    pdp = PonpareDataPreprocessing('ponpare', 'ponpareDB')
    
    user_list = pdp.loadUserListData()
    user_list_age_cat = pdp.generateAgeCategoryForUsers(user_list)
    pdp.storeUserInfoInDB(user_list_age_cat)
       
    coupon_list = pdp.loadCouponListTrainData()
    coupon_list_price_cat = pdp.generatePriceCategoryForCoupons(coupon_list)
    coupon_list_all_cat  = pdp.generateDiscoutCategoryForCoupons(coupon_list_price_cat)
    pdp.storeCouponInfoInDB(coupon_list_all_cat)
    
    coupon_id_2_cluster_id = pdp.createCouponId2ClusterId(coupon_list_all_cat)
    pdp.storeCouponId2ClusterIdInDB(coupon_id_2_cluster_id)
    
    coupon_clust_def = pdp.createCouponClusterDef(coupon_list_all_cat)
    pdp.storeCouponClustDefInDB(coupon_clust_def)
       
    coupon_visit = pdp.loadCouponVisitTrainData()    
    pruned_coupon_visit = pdp.pruneCouponVisitTrain(coupon_visit, coupon_list)
    pruned_coupon_cluster_visit = pdp.addClusterIdPrunedCouponVisit(pruned_coupon_visit, coupon_id_2_cluster_id)
    pdp.storeUserCouponActivityInfo(pruned_coupon_cluster_visit)

