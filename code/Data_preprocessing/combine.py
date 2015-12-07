#Team: 5DB Minds
#for combining 12 csv files into

import pandas as pd
import os
import time

start_time = time.time()
app_attributes = "AppSize, Category, ContentRating, Description, Developer, HaveInAppPurchases, Instalations, IsFree, IsTopDeveloper, LastUpdateDate, Name, Price, PublicationDate, Score"
input_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/csv/'
out_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/'

filelist = os.listdir(input_dir)

fout=open(out_dir+"out.csv","w")
fout.write(str(app_attributes))
fout.close()

for file_ in filelist:
    	print file_
	fout=open(out_dir+"out.csv","a")
	isFirstLine = 1
	for line in open(input_dir+file_):
		if isFirstLine == 1:
			isFirstLine = 0
			pass
		else:
			fout.write(line)

fout.close() 

df = pd.read_csv(out_dir+"out.csv")
print df.shape

print df.head()
'''for num in range(2,201):
    f = open("sh"+str(num)+".csv")
    f.next() # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()


print df.shape
df.columns = header # final sequential column name
df.to_csv(out_dir+'big_data9.csv',encoding='utf-8')'''
#df.to_csv('big_data.csv',encoding='utf-8',header=header, sep=',')
print("--- %s seconds ---" % (time.time() - start_time))



