#Team: 5DB Minds
# Transform data

import pandas as pd
import os
import time
import numpy as np
import sklearn
import AppVectorizerModule as avm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from  datetime import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

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
input_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/csv/big_data1.csv'

df = pd.DataFrame()
df = pd.read_csv(input_dir, index_col=False)
df = df.drop(df.columns[0], axis=1)# drop the column of row index
print df.shape
column_to_drop = ['Description', 'Developer', 'Name']
label = df['Score'] # label of train data
# AppSize Price - Already Numerical
# 'Category', 'Developer', 'Name', 'ContentRating' - Text Vectorization

df_train = df[:]
#df_train = df_train.drop('Description', 1)
#df_train = df_train.drop('Developer', 1)
#df_train = df_train.drop('Name', 1)

#----------------------------- Replace binary---------------------------------------#
# Replace binary features having 0 and 1 (IsTopDeveloper, HaveInAppPurchase, IsFree)
df_train = df_train.replace([False, True],['0','1'])

#---------------------- Replace other text attributes-------------------------------#
# Replace features needing other kinds of transformation PublicationDate, LastUpdateDate, Instalations
df_train.PublicationDate=map(mapDates,df_train.PublicationDate)
df_train.LastUpdateDate=map(mapDates,df_train.LastUpdateDate)
df_train.Instalations = map(mapInstalation, df_train.Instalations)
#df.Score=map(mappingScores,df.Score)
#df.AppSize=map(processAppSize,df.AppSize)

#-------------------------split 100K points points into 20 segments------------------#
'''splitted = [pd.DataFrame() for x in xrange(20)]

splitted[0] = df_train[0:5000]
for i in range(1, 20):
	j = 5000
	splitted[i] = df_train[j*i:j*(i+1)]
'''
#----------------------------------Vectorization--------------------------------------#
min_df = 1
max_df = 0.99 #it's value lies in: [0.7, 1.0), remove the word that occur in more than 90% of all the posts.
token_pattern = r"\b[a-z]\b"
lowercase = True
stem_vectorizer = avm.StemmedCountVectorizer(encoding='utf-8',
                                         min_df =min_df,
                                         max_df =max_df,
                                         stop_words='english',
                                         analyzer='word',
                                         lowercase = lowercase)
##set filterparameter to your vectorizer
filter_by=["OnlyEng", "AllLang"] #two options are available
count_dialect = True 
#n_samples = n_samples #as u choose it.
stem_vectorizer.setfilter_option(filter_by[0],count_dialect)

#df_train = df_train[:100000]
df_train = df_train[:75000]
df_train = df_train.sample(frac=0.04)

vector_df1, fmatrix, column_vectorizer = avm.vectorize_columnTfIdf(df_train, 'Category',vectorizer=stem_vectorizer, tf_idf=True)
#vector_df2, fmatrix, column_vectorizer = avm.vectorize_columnTfIdf(df_train, 'Developer',vectorizer=stem_vectorizer, tf_idf=True)
#vector_df3, fmatrix, column_vectorizer = avm.vectorize_columnTfIdf(df_train, 'Name',vectorizer=stem_vectorizer, tf_idf=True)
vector_df4, fmatrix, column_vectorizer = avm.vectorize_columnTfIdf(df_train, 'ContentRating',vectorizer=stem_vectorizer, tf_idf=True)
#vector_df5, fmatrix, column_vectorizer = avm.vectorize_columnTfIdf(df_train, 'Description',vectorizer=stem_vectorizer, tf_idf=True)

#---------------------------------------PCA-----------------------------------------#
#dist = pdist(vector_df, 'euclidean')
#print dist.shape
#dist = squareform(dist)
#print dist.shape

#pca = PCA(n_components=10, copy=True)
#vector_df2_pca = pd.DataFrame(pca.fit(vector_df2).transform(vector_df2))
#vector_df3_pca = pd.DataFrame(pca.fit(vector_df3).transform(vector_df3))
#vector_df5_pca = pd.DataFrame(pca.fit(vector_df5).transform(vector_df5))
#print 'finished pca'
#print vector_df2_pca.shape
#print vector_df3_pca.shape
#---------------------------------------TSNE-----------------------------------------#
#tsne = TSNE(n_components=3, init='pca', random_state=0)
#tsne = TSNE(n_components=2,perplexity=1,early_exaggeration=1,learning_rate=100,n_iter=200, /
#random_state=0,metric='precomputed',init='random')
#Y = tsne.fit_transform(vector_df)

df_train = df_train.drop('Category', 1)
df_train = df_train.drop('Developer', 1)
df_train = df_train.drop('Name', 1)
df_train = df_train.drop('ContentRating', 1)
df_train = df_train.drop('Description', 1)

#df_train = pd.concat([df_train, vector_df1], axis=1, ignore_index=True)
#df_train = pd.concat([df_train, vector_df2_pca], axis=1, join_axes=[df_train.index])
#df_train = pd.concat([df_train, vector_df3_pca], axis=1, join_axes=[df_train.index])
#df_train = pd.concat([df_train, vector_df4], axis=1, ignore_index=True)
#df_train = pd.concat([df_train, vector_df5_pca], axis=1, join_axes=[df_train.index])
#df = pd.concat([df_train, vector_df1, vector_df4], axis=1, join_axes=[df_train.index])

'''for cols in vector_df1.columns:
	s1 = vector_df1[cols]
	df_train[cols] = pd.Series(s1, index=df_train.index)'''

#arr = df['ContentRating'].as_matrix()
#print np.unique(arr).size
#print np.unique(arr)
#print df_train.head()

out_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/numerical_csv/'

print df_train.shape
df_train.to_csv(out_dir+'big_data_with_2_vec_1_3K.csv')
vector_df1.to_csv(out_dir+'vector_df1.csv')
vector_df4.to_csv(out_dir+'vector_df4.csv')

#vector_df2.to_csv(out_dir+'Developer.csv',encoding='utf-8')
#vector_df5.to_csv(out_dir+'Description.csv')

print("--- %s seconds ---" % (time.time() - start_time))











