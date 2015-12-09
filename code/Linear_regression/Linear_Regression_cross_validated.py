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
input_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/numerical_csv/3K/big_data_without_vec_1_3K_9features.csv'
#big_data_vec_1_3K_pca_34features
#big_data_with_2_vec_1_3K_34features
#big_data_without_vec_1_3K_9features
df = pd.DataFrame()
df = pd.read_csv(input_dir)
print df.head()
df = df.drop(df.columns[0], axis=1)# drop the column of row index
print df.shape

data_points = 10000

lr = LinearRegression()

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validated:
y = df['Score']
y = y.fillna(0)
y = y.as_matrix()
df = df.drop('Score', axis=1)
print df.head()


predicted = cross_val_predict(lr, df.as_matrix(), y, cv=10)
print df.columns

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1, marker='+')
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))




