#Team: 5DB Minds
# Transform data

import pandas as pd
import os
import time
import numpy as np
import sklearn
from  datetime import datetime

def mapDates(d):
    d=d[:10]
    d1= datetime.now()
    d2=datetime.strptime(d, "%Y-%m-%d")
    return (d1.year - d2.year)*12 + d1.month - d2.month 

def mapInstalation(x):
	if x == '1 - 5':
		return 1
	elif x == '5 - 10':
		return 2
	elif x == '10 - 50':
		return 10
	elif x == '50 - 100':
		return 20
	elif x == '100 - 500':
		return 100
	elif x == '500 - 1,000':
		return 200
	elif x == '1,000 - 5,000':
		return 1000
	elif x == '5,000 - 10,000':
		return 2000
	elif x == '10,000 - 50,000':
		return 10000
	elif x == '50,000 - 100,000':
		return 20000
	elif x == '100,000 - 500,000':
		return 100000
	elif x == '500,000 - 1,000,000':
		return 200000
	elif x == '1,000,000 - 5,000,000':
		return 1000000
	elif x == '5,000,000 - 10,000,000':
		return 2000000
	elif x == '10,000,000 - 50,000,000':
		return 10000000
	elif x == '50,000,000 - 100,000,000':
		return 20000000
	else:
		return 0

# Start
start_time = time.time()
input_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/csv/big_data2.csv'

df = pd.DataFrame()
df = pd.read_csv(input_dir, index_col=False)
df = df.drop(df.columns[0], axis=1)# drop the column of row index
#print df.head()

column_to_drop = ['Description', 'Category', 'Developer', 'Name', 'ContentRating']
label = df['Score'] # label of train data
# AppSize # Price

df_train = df.drop(column_to_drop, 1)
print df_train.shape

# Replace binary features having 0 and 1 (IsTopDeveloper, HaveInAppPurchase, IsFree)
df_train = df_train.replace([False, True],['0','1'])

# Replace features needing other kinds of transformation PublicationDate, LastUpdateDate, Instalations
#df.Score=map(mappingScores,df.Score)
df_train.PublicationDate=map(mapDates,df_train.PublicationDate)
df_train.LastUpdateDate=map(mapDates,df_train.LastUpdateDate)
df_train.Instalations = map(mapInstalation, df_train.Instalations)
#df.AppSize=map(processAppSize,df.AppSize)

#arr = df['Instalations'].as_matrix()
#print np.unique(arr).size
#print np.unique(arr)

print df_train.head()

df_train.to_csv('big_data2.csv')

print("--- %s seconds ---" % (time.time() - start_time))











