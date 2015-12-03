#Team: 5DB Minds
#linear regression
import pandas as pd
import os
import time

start_time = time.time()
input_dir = '/media/naila/New Volume/CSE_6990_Big_Data_and_Data_Science/Project/data/out'
out_dir = 'big_csv.csv'

if not os.path.exists(out_dir): os.makedirs(out_dir)

filelist = os.listdir(input_dir)
df = pd.DataFrame(columns=app_attributes)

for file_ in filelist:
    	print file_
	fout=open("out.csv","a")
# first file:
for line in open("sh1.csv"):
    fout.write(line)
# now the rest:    
for num in range(2,201):
    f = open("sh"+str(num)+".csv")
    f.next() # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()


print df.shape
df.columns = header # final sequential column name
df.to_csv(out_dir+'big_data9.csv',encoding='utf-8')
#df.to_csv('big_data.csv',encoding='utf-8',header=header, sep=',')
print("--- %s seconds ---" % (time.time() - start_time))



