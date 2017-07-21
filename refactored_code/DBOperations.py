# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:57:03 2017

@author: tiwarir
"""

import sqlite3
import pandas as pd

class DBOperations(object):
    def __init__(self, dbName):
        self.conn = sqlite3.connect(dbName)
        self.conn.text_factory = str
        self.cursor = self.conn.cursor()

    #def __del__(self):
    #   self.commitAndClose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.commitAndClose()

    def execute(self, sqlPrepStmt, sqlArgsList = None, isSilent = None):
        isDbOpSuccess = True
        records = []
        try:
            if sqlArgsList is None:
                results = self.cursor.execute(sqlPrepStmt)
                records = [record for record in results]
            else:
                results = self.cursor.execute(sqlPrepStmt, sqlArgsList)
                records = [record for record in results]

            records = [record[0] for record in records] # convert from list of tuples to just list
        except sqlite3.OperationalError, msg:
            if isSilent is not None:
                print 'Error: OperationalError Occurred - Error Msg: ', msg
            isDbOpSuccess = False
        except sqlite3.IntegrityError, msg:
            if isSilent is not None:
                print 'Error: IntegrityError Occurred - Error Msg: ', msg
            isDbOpSuccess = False
        return records, isDbOpSuccess

    def commitData(self):
        self.conn.commit()

    def closeConnection(self):
        self.conn.close()

    def commitAndClose(self):
        self.commitData()
        self.closeConnection()

    def getConnection(self):
        return self.conn

    def getAllRecords(self, tableName):
        records = []
        try:
            sql = 'SELECT * FROM ' + tableName
            fetchedRecords = self.cursor.execute(sql)
            records = [record for record in fetchedRecords]
        except sqlite3.OperationalError, msg:
            print ('Error Occurred - ',
                'Select Query Failed! Msg: ', msg)
            return -1
        return records

    def getColumnNames(self, tableName):
        sql = 'SELECT * FROM ' + tableName
        cur = self.getConnection().execute(sql)
        column_names = list(map(lambda x: x[0], cur.description))
        return column_names

    def get_all_records_as_df(self, table_name):
        loaded_data = self.getAllRecords(table_name)
        column_names = self.getColumnNames(table_name)
        records_df = pd.DataFrame(columns=column_names, data=loaded_data)
        return records_df

    def conditional_selection(self, table_name, column_name, value):
        query = 'SELECT * FROM ' + table_name + ' WHERE ' + column_name + '=' + str(value)
        df = pd.read_sql_query(query, self.conn)
        return df



###############################################################################
if __name__ == '__main__':
    dbo = DBOperations('ponpareDB')
    coupon_clust = dbo.conditional_selection('CouponClustInfo', 'PRICE_CATEGORY','"catalog_price1"')
    user_activity = dbo.conditional_selection('UserCouponActivityInfo', 'PURCHASE_FLG', 1)
    

