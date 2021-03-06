#Team: 5DB Minds
# Naila Bushra
# Linear regression

import pandas as pd
import os
import time
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cross_validation import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

# normalize 1
def normalize(df):
    # iterate over columns
    for cols in df.columns:
	num = df[cols]
	# exclude columns having -1 and 0
        if np.sum(num) == len(num)*(-1) or np.sum(num) == 0:
                pass
	else:
		df[cols] = (df[cols] - df[cols].mean())/df[cols].std()

    return df

start_time = time.time()
input_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/numerical_csv/3K/big_data_without_vec_1_3K_9features.csv'
#big_data_vec_1_3K_pca_34features
#big_data_with_2_vec_1_3K_34features
#big_data_without_vec_1_3K_9features

df = pd.DataFrame()
df = pd.read_csv(input_dir)
df = df.drop(df.columns[0], axis=1)# drop the column of row index
print df.shape
#print df.head()

#y = df['Score']
#df = df.drop('Score', axis=1)

#pca = PCA(n_components=25, copy=True)
#df = pd.DataFrame(pca.fit(df).transform(df))

#tsne = TSNE(n_components=3, init='pca', random_state=0)
#Y = pd.DataFrame(tsne.fit_transform(df))

#df['Score'] = y
print df.head()

#df = df.drop('0.4', axis=1)
#df = df.drop('4.3', axis=1)
#df = df.drop('3.2', axis=1)
#df = df.drop('2.1', axis=1)
#df = df.drop('3.3', axis=1)

kf = KFold(3000, n_folds=4)
for train, test in kf:
	#print 'train=',train
	#print 'test=',test

	df_train = df.loc[train]
	df_test = df.loc[test]

	#print 'train shape', df_train.shape
	#print 'test shape', df_test.shape

	label_train = df_train.as_matrix(columns=['Score'])
	label_test = df_test.as_matrix(columns=['Score'])

	df_train = df_train.drop('Score', axis=1)
	df_test = df_test.drop('Score', axis=1)
	#lr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=4)
	lr = LinearRegression()

	lr.fit(df_train.as_matrix(), label_train)

	# The coefficients
	#print('Coefficients: \n', lr.coef_)

	# The mean square error
	print("Residual sum of squares: %.2f"%np.mean((lr.predict(df_test.as_matrix())-label_test)**2))

	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % lr.score(df_test, label_test))

	# Plot outputs
	#plt.scatter(df_test.columns[0], label_test,  color='black')
	#plt.plot(df_test.columns[0], lr.predict(df_test), color='blue', linewidth=3)

	#plt.xticks(())
	#plt.yticks(())

	#plt.show()

print("--- %s seconds ---" % (time.time() - start_time))




