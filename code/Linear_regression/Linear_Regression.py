#Team: 5DB Minds
# Naila Bushra
# Linear regression

import pandas as pd
import os
import time
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

start_time = time.time()
input_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/numerical_csv/big_data_vec_1_5K.csv'

df = pd.DataFrame()
df = pd.read_csv(input_dir)
df = df.drop(df.columns[0], axis=1)# drop the column of row index
print df.shape

data_points = 5000
# k fold
#df_train = df[:3500]
#df_test = df[3500:]
#label_train = df_train.as_matrix(columns=['Score'])
#label_test = df_test.as_matrix(columns=['Score'])

kf = KFold(5000, n_folds=10)
for train, test in kf:
	#print 'train=',train
	#print 'test=',test

	df_train = df.loc[train]
	df_test = df.loc[test]

	print 'train shape', df_train.shape
	print 'test shape', df_test.shape

	label_train = df_train.as_matrix(columns=['Score'])
	label_test = df_test.as_matrix(columns=['Score'])

	#lr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=4)
	lr = LinearRegression()

	lr.fit(df_train, label_train)

	# The coefficients
	#print('Coefficients: \n', lr.coef_)

	# The mean square error
	print("Residual sum of squares: %.2f"% np.mean((lr.predict(df_test.as_matrix()) - label_test) ** 2))

	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % lr.score(df_test, label_test))

	# Plot outputs
	'''plt.scatter(df_test, label_test,  color='black')
	plt.plot(df_test, lr.predict(df_test), color='blue', linewidth=3)

	plt.xticks(())
	plt.yticks(())

	plt.show()'''

print("--- %s seconds ---" % (time.time() - start_time))




