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

start_time = time.time()
input_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/numerical_csv/big_data1.csv'

df = pd.DataFrame()
df = pd.read_csv(input_dir)
df = df.drop(df.columns[0], axis=1)# drop the column of row index
print df.shape

# k fold
df_train = df[:70000]
df_test = df[70000:]

label_train = df_train.as_matrix(columns=['Score'])
label_test = df_test.as_matrix(columns=['Score'])

#df_train, df_test, label_train, label_test = cross_validation.train_test_split(\
#df, df['Score'], test_size=0.3, random_state=0)
print df_train.shape, label_train.shape
print df_test.shape, label_test.shape
 
#lr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=4)
lr = LinearRegression()

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validated:
y = df['Score'].as_matrix()
predicted = cross_val_predict(lr, df.as_matrix(), y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1, marker='+')
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

lr.fit(df_train, label_train)

# The coefficients
print('Coefficients: \n', lr.coef_)

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




