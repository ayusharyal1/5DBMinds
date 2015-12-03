#Team: 5DB Minds
# Naila Bushra
# Transform data

import pandas as pd
import os
import time
import numpy as np
import sklearn

start_time = time.time()
input_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/csv/big_data2.csv'

df = pd.DataFrame()
df = pd.read_csv(input_dir, index_col=False)
df = df.drop(df.columns[0], axis=1)# drop the column of row index
print df.head()

column_to_drop = ['Description', 'Category', 'Developer', 'Name', 'ContentRating']
label = df['Score'] # labe;=l oftrain data
# AppSize
# Price
# IsTopDeveloper
# HaveInAppPurchase
# IsFree
#'PublicationDate',
#'LastUpdateDate' 
#'Instalations',

df_train = df.drop(column_to_drop, 1)
print df_train.shape

#arr = df['ContentRating'].as_matrix()
#print np.unique(arr).size
#print np.unique(arr)

# Replace binary features having 0 and 1 (IsTopDeveloper, HaveInAppPurchase, IsFree)
df_train = df_train.replace([False, True],['0','1'])
print df_train.head()

# Replace features needing other kinds of transformation PublicationDate, LastUpdateDate, Instalations
# do the transformation

df_train.to_csv('big_data2.csv')

print("--- %s seconds ---" % (time.time() - start_time))

