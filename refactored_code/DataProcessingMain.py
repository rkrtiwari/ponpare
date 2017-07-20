#!/usr/bin/python
"""
Date: Jun 09 2017
Description: Refactored code for CSL
@author: rsugumar
"""

#
#usage: python DataProcessingMain.py [-h] [-d <DATA>] -p PATH [-db DBNAME]
#   -d <DATA>           Data Input Format. Default: PONPARE (Optional arg)
#   -p <Data path>      Path of the Input Data.
#   -db <DB Name>       Name of the DB file to store the data. Default: ponpareInfo.db  (Optional arg)
#

import argparse
from PonpareDataPreprocessing import PonpareDataPreprocessing

def main():
    parser = argparse.ArgumentParser('DataProcessingMain.py', description='CSL\'s Data Preprocessing Step', epilog='''
        This procedure extracts the necessary data from the given data path, 
        and does data preprocessing for CSL to process. Necessary procedure.''')
    parser.add_argument('-d', '--data', help = 'Input Data Format - PONPARE', default = 'PONPARE', required = False)
    parser.add_argument('-p', '--path', help = 'Input Data Path', default = 'ponpareData', required = True)
    parser.add_argument('-db', '--dbname', help = 'Name of the output DB', default = 'ponpareInfo.db', required = False)

    args = vars(parser.parse_args())
    dataFormat = args['data']
    dataPath = args['path']
    outputDbName = args['dbname']

    if dataFormat.lower() == 'ponpare':
        ponpareDP = PonpareDataPreprocessing(dataPath, outputDbName)
        ponpareDP.startProcess()
    else:
        print dataFormat, ': Unsupported format! Options: PONPARE'

if __name__ == '__main__':
    main()
