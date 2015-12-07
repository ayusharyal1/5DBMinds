#Team: 5DB Minds
# Naila Bushra
#Converts json files to csv

import pandas as pd
import os
import json
import unicodedata
import time

start_time = time.time()
input_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/smallDataChunksDivided/9/'
app_attributes = ['AppSize', 'Category', 'ContentRating', 'Description', 'Developer', 'HaveInAppPurchases', 'Instalations', 'IsFree', 'IsTopDeveloper', 'LastUpdateDate', 'Name', 'Price', 'PublicationDate', 'Score']
#'Reviewers', 'Reviews', 'ReviewsStatus', 'Score']
out_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/out/'
filelist = os.listdir(input_dir)
df = pd.DataFrame(columns=app_attributes)

# not used, converts unicode to string
def utos(uni):
    return unicodedata.normalize('NFKD', uni).encode('ascii','ignore')

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
#df.to_csv('big_data.csv',encoding='utf-8',header=header, sep=',')
print("--- %s seconds ---" % (time.time() - start_time))



