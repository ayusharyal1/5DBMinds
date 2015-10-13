#Team: 5DB Minds
#We have splitted the 6.2 GB JSON file into 1146 files for data handling convenience. The command for splitting is
#split -l 1000 PlayStore_2015_07.json <dir_where_to_be_splitted>

​import pandas as pd
import os

input_dir = '../smallest/'

#read them into pandas
filelist = os.listdir(input_data)
df = pd.DataFrame()

for file_ in filelist:
    print file_
    # read the entire file into a python array
    with open(input_dir+file_, 'rb') as f:
        line = f.readlines()
    # remove the trailing "\n" from each line
    line = map(lambda x: x.rstrip(), line)
    print line
    print type(line)
    #tmp = pd.read_json(input_dir+file_)
    #print tmp.head()
	
    #df = df.append(tmp)
