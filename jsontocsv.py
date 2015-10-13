#Team: 5DB Minds
#We have splitted the 6.2 GB JSON file into 1146 files for data handling convenience. The command for splitting is
#split -l 1000 PlayStore_2015_07.json dir_where_to_be_splitted
import pandas as pd
import os
import json
import pprint
import csv
import unicodedata
import ast

input_dir = './smallest/'
app_attributes = ['AppSize', 'Category', 'ContentRating', 'Developer', 'Description', 'HaveInAppPurchases', \
'Instalations', 'IsFree', 'IsTopDeveloper', 'LastUpdateDate', 'Name', 'Price', 'PublicationDate', 'Reviewers', \
'Reviews', 'ReviewsStatus', 'Score']
out_dir = './out/'
filelist = os.listdir(input_dir)
df = pd.DataFrame(index=None, columns=app_attributes)

def utos(uni):
    return unicodedata.normalize('NFKD', uni).encode('ascii','ignore')

if not os.path.exists(out_dir): os.makedirs(out_dir)

for file_ in filelist:
    with open(input_dir+'data', 'rb') as f:
        line = f.readlines()
	# remove the trailing "\n" from each line
	line = map(lambda x: x.rstrip(), line)
    print len(line)

    for i in range(0,len(line)):
	#json_acceptable_string = line[i].replace("'", "\"")
        dn = json.loads(line[i])
	for key,value in dn.items():
		if key in df.columns:
			print dn[key]
			df[key]=dn[key]

    #for col in df.columns:
	#df[col].append(dn["AppSize"])
    #df = df.append(line)

#df = df[1:]
print df.head()
print df.shape

#df.to_csv('big_data.csv')
df.to_json('big_data.json')
'''df = pd.read_json('big_data.json')
f = open('big_data.json')  
data = json.load(f)
f.close()

print data.shape
print type(data)'''
