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

start_time = time.time()
input_dir = 'big_data1.csv'

df = pd.DataFrame()
df = pd.read_csv(input_dir)
df = df.drop(df.columns[0], axis=1)# drop the column of row index

df_test = pd.DataFrame()
df_test = pd.read_csv('big_data2.csv')
df_test = df_test.drop(df_test.columns[0], axis=1)# drop the column of row index

label = df['Score']
label_test = df_test['Score']
#df = df.drop(df['Score'], axis=1)# drop the column of row index

lr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=4)
lr.fit(df.as_matrix(), df['Score'])

# The coefficients
print('Coefficients: \n', lr.coef_)

# The mean square error
print("Residual sum of squares: %.2f"% np.mean((lr.predict(df_test.as_matrix()) - diabetes_y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lr.score(diabetes_X_test, diabetes_y_test))

'''predicted = cross_val_predict(lr, df_train, y, cv=1000)

print predicted.shape
# Plot outputs
plt.scatter(df['Score'], predicted,  color='black')
plt.plot(df['Score'], lr.predict(df['Score']), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
'''

'''
fig,ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
fig.show()
'''
'''figure()
plot(k, var, 'r')
xlabel('k')
ylabel('var')
title('var vs k')
show()'''
print("--- %s seconds ---" % (time.time() - start_time))

'''
app_attributes = ['AppSize', 'Category', 'ContentRating', 'Description', 'Developer', 'HaveInAppPurchases', 'Instalations', 'IsFree', 'IsTopDeveloper', 'LastUpdateDate', 'Name', 'Price', 'PublicationDate', 'Score']
#'Reviewers', 'Reviews', 'ReviewsStatus', 'Score']

filelist = os.listdir(input_dir)
df = pd.DataFrame(columns=app_attributes)

if not os.path.exists(out_dir): os.makedirs(out_dir)

row_count = 0;
for file_ in filelist:
    	print file_
	with open(input_dir+file_, 'rb') as f:
		line = f.readlines()
		# remove the trailing "\n" from each line
		line = map(lambda x: x.rstrip(), line)
    	#print len(line)
	row = [] # dataframe row
	header = [] # dataframe header
    	for i in range(0,len(line)):
		dn = json.loads(line[i])
		header = []
		for key,value in dn.items():
			if key in app_attributes:# our selected feature attributes
				if key == 'LastUpdateDate' or key == 'PublicationDate':# nested values
					value = value['$date']
				elif	key == 'Score':# nested values
					value = value['Total']
				#if type(value) == 'unicode':
					#value = utos(value)
				row.append(value) # make list of attribute values
				header.append(key) # make list of column names
		# add rows in dataframe
		df.loc[row_count] = row
		row = []
		row_count = row_count + 1
print df.shape
df.columns = header # final sequential column name
df.to_csv(out_dir+'big_data9.csv',encoding='utf-8')
#df.to_csv('big_data.csv',encoding='utf-8',header=header, sep=',')'''



