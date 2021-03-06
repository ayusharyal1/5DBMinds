
# coding: utf-8

# In[11]:


#Converting categorical data into numbers with Pandas and Scikit-learn
#feature extraction. 
#When it involves a lot of manual work, this is often referred to as feature engineering.


# In[228]:

import numpy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import sys


# In[217]:

import pandas as pd
from pandas import *
from numpy import *
import numpy as np
import os
from pandas import DataFrame
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import scipy as sp
import nltk.stem
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import string
from collections import Counter
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer


# In[218]:

app_file = '../data/big-data-csv.csv'
appdf = pd.read_csv(app_file,sep=',')
appdf.head(2)


# In[41]:

col_cat = appdf.Category
col_cat.head(2)


# In[42]:

#Total Number of Columns:
print("n_samples"),max(appdf.index)+1


# In[43]:

appdf.columns.values.tolist()


# In[44]:

len(col_cat.unique())
col_cat.unique()


# In[134]:

class LemmaTokenizer(object):
    
    def __init__(self):
        
        self.wnl = WordNetLemmatizer()
        
    def __call__(self, doc):
        
        lowers = doc.lower()
        doc = lowers.translate(None, string.punctuation) ##remove the punctuation using the character
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

#example, vect = CountVectorizer(tokenizer=LemmaTokenizer()) 

best_doc = None
best_i = None

'''Computes eculidean distance between two normalized vectors v1 and v2'''
def dist_norm(v1,v2):
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    delta= v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray()) #norm() calculates the Eculidean norm i.e. shortest distance"

def best_match(column_vectorizer, fmatrix,text_to_compare):
    n_samples = 100 # fmatrix.shape[0]
    best_dist = sys.maxint
    vect_to_compare = column_vectorizer.transform(text_to_compare)
    for i in range(0, n_samples):
        text_in_column = col_cat[i]
        if text_in_column == text_to_compare[0]:
            continue
        vector_for_column_text = fmatrix.getrow(i)
        #d = dist_raw(post_vec, new_post_vec)
        d = dist_norm(vector_for_column_text, vect_to_compare)
        print "===Category of app- %i with dist = %.2f: %s"%(i,d,text_in_column)
        if d < best_dist:
            best_dist = d
            best_i = i
    print "Best text in category is %i with dist = %.4f"%(best_i,best_dist)
print type([col_cat[4]])
#best_match(column_vectorizer,fmatrix, [col_cat[4]])


# In[178]:

'''Use StemmedCountVectorizer to do:
1. lower casing the raw post in the preprossing step done in parent calss.
2. Extracting all individual words in the tokenization step in parent class.
3. Converting each word into its stemmed version.'''

class StemmedCountVectorizer(CountVectorizer):

    ##overiding the analyzer of CountVectorizer
    def build_analyzer(self):
        english_stemmer = nltk.stem.SnowballStemmer('english')
        analyzer = super(StemmedCountVectorizer,self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

def stat_vectorized_matrix(vectorized_array,vectorizer_type=None):
   
    #count the number of features generated,
    m, n = vectorized_array.shape
    count_non_zero_cells = np.count_nonzero(vectorized_array) #vectorized_array.nnz
    print("vectorizer_type:"),type(column_vectorizer)
    print("Sparse matrix shape: "), vectorized_array.shape
    #count the number of non-zero entries,
    print("Sparsity(%%of non-zero values): %.6f %%" %(count_non_zero_cells/float(m*n) * 100))


# In[98]:

stopwords_list = stopwords.words('english')


# ### How to Install Stop Words ?
# 
#     In your terminal:
#                     $python
#                     >>import nltk
#                     >>nltk.download()
#                     >>d ##hit 'd'
#                     >>stopwords ##type stopwords
#     Suppose you downloaded the stopwords in your '~/nltk_data/corpora/stopwords' folder
#             1. Perhaps the folder is downloaded in your current directory ~/nltk_data/corpora/stopwords
#             2. Extract 5DBMinds/data/stopwords-extended.zip of our project repository in github.
#             3. Copy all files to your ~/nltk_data/corpora/stopwords folder.

# In[224]:

'''Returns a vectorized ND dataframe, vectorized ndarray, 
and an instance of the vectorizer Class used to transform.'''

def vectorize_column(dataframe,column_name,vectorizer=None):
    if vectorizer is None:
        print("No Vectorizer is explicitly specified. Using CountVectorizer as default one. ")
        column_vectorizer = CountVectorizer(min_df=1)
    else:
        column_vectorizer = vectorizer
    
    if column_name in dataframe.columns.values.tolist():
        column_df = dataframe[column_name] #select all the samples from the column passed as param.
        fmatrix = column_vectorizer.fit_transform(column_df) #convert text features to numerical vectors
        dataframe_f = pd.DataFrame(fmatrix.toarray(), columns=column_vectorizer.get_feature_names())
        print("Dataframe shape :("),dataframe_f.index.max()+1,",", dataframe_f.head(1).shape[1],")"
        
        return dataframe_f, fmatrix, column_vectorizer
    else:
        print("No column found")
        
        
'''Returns a vectorized (n_samples,n_features) dataframe, matrix and vectorizing object.
Parameters:
dataframe: pandas dataframe object
column_name: name of the column you want to vectorize (a column in above dataframe object)
vectorizer= Vectorizer Object, if none then CountVectorizer is used as default. 
n_samples: number of rows you want to vectorize
tf_idf: if True then TF-IDF matrix is returned, else only matrix of term frequency is return.

USAGE:
stem_vectorizer = StemmedCountVectorizer(encoding='utf-8',
                                         min_df =min_df,
                                         max_df =max_df,
                                         stop_words='english',
                                         analyzer='word',
                                         lowercase = lowercase)
dfx, matrixX, sv = vectorize_columnTfIdf(df, 'my_column',vectorizer=stem_vectorizer, n_samples=100, tf_idf=True)
'''

def vectorize_columnTfIdf(dataframe,column_name,vectorizer=None, n_samples=None, tf_idf=False):
    
    more_stopwords  = ['00','000','0000','0003','0004','0004','0005'] 
    more_stopwords += stopwords.words('english')
    more_stopwords += stopwords.words('japanese') 
    more_stopwords += stopwords.words('chinese')
    more_stopwords += stopwords.words('arabic')
    more_stopwords += stopwords.words('korean')
    more_stopwords += stopwords.words('russian')    
    
    if vectorizer is None:
        print("No Vectorizer is explicitly specified. Using CountVectorizer as default one. ")
        column_vectorizer = CountVectorizer(min_df=1) #default vectorizer
    else:
        column_vectorizer = vectorizer
        column_vectorizer.stop_words = more_stopwords
    
    if column_name in dataframe.columns.values.tolist():
        
        if n_samples is None:
            column_df = dataframe[column_name] #select all the samples from the column passed as param. 
            print len(column_df)
        else:
            column_df = dataframe[column_name].iloc[0:n_samples] #select all the samples from the column passed as param.
            print len(column_df)
        
        fmatrix = column_vectorizer.fit_transform(column_df)   
        
        if(tf_idf is True):
            
            tfidf_transformer  = TfidfTransformer(norm='l2').fit(fmatrix)
            tfidfNormalzedmatrix = tfidf_transformer.transform(fmatrix)
            fmatrix = tfidfNormalzedmatrix
            
        dataframe_f = pd.DataFrame(fmatrix.todense(), columns=column_vectorizer.get_feature_names())
        print("formed dataframe of size:("),dataframe_f.index.max()+1,",", dataframe_f.head(1).shape[1],")"
        
        return dataframe_f, fmatrix, column_vectorizer
    else:
        print("No column found")


# ### Remove following words:
# 
#     5. Do capital letters carry information? [Lowercasing]
#     4. Does distinguishing inflected form ("goes" vs. "go") carry information?[Stemming/Lemmantizing]
#     3. Do interjections, determiners carry information (Stop Words)?
#     2. Does numerical strings carries information? 000, 000, 100 	000, 0000,000031,0002, 03 ,004,	0005 	0006 	0007
#     1. 
# ####  Term Frequency: 
#     Counting how many times does a word occur in each message (Term Freq.)
# #### Inverse  Document Frequency:
#     weighting the counts, so that frequent tokens get lower weight 
# #### Normalization
#     normalizing the vectors to unit length, to abstract from the original text length (L2 Norm)

# ### CATEGORY
# #### '''Analysis of 'Category' data-columns
# ##### Each of the application has only one 'category' so the each of the category is equi-distance from all other.
# Though similarity of each of the values of category is same, the category-name itself might not effect the rating equally.
# That is why they are inluded as training features.'''

# In[133]:

'''Possible type of count vectorizer that could be used.
    Examples: 
    column_vectorizer = CountVectorizer(min_df=1)
    column_vectorizer = CountVectorizer(min_df =1, stop_words='english') 
    print column_vectorizer.get_feature_names()
    Do not ASSIGN max_df and min_df if you are using TF-IDF. Because tf-idf considers the case.
'''
min_df = 1
max_df = 0.99 #it's value lies in: [0.7, 1.0), remove the word that occur in more than 90% of all the posts.
token_pattern = r"\b[a-z]\b"
lowercase = True
stem_vectorizer = StemmedCountVectorizer(encoding='utf-8',
                                         min_df =min_df,
                                         max_df =max_df,
                                         stop_words='english',
                                         analyzer='word',
                                         lowercase = lowercase)
print("CATEGORY")
cat_newfeature, cat_fmatrix, cat_column_vectorizer = vectorize_column(appdf, 'Category', stem_vectorizer)
stat_vectorized_matrix(cat_fmatrix.toarray(), column_vectorizer)
#print len(column_vectorizer.vocabulary_) #vocabulary_ is feature set.
cat_newfeature.head(3)


# ### Use TD-IDF Transformer, Also
#     1. Evaluate change in sparcity. No change in sparcity.

# In[223]:

cat_tfidf_transformer  = TfidfTransformer().fit(cat_fmatrix)
cat_nd_array_x = cat_tfidf_transformer.transform(cat_fmatrix.toarray(), copy=True)


# In[125]:

stat_vectorized_matrix(cat_nd_array_x.toarray(),cat_tfidf_transformer)

print nd_array_x.toarray()[0:10,0:10]


# #### #Analysis of Description Field
# 

# In[220]:

#stopwords = stopwords.words('english') #37386, 37225,37185

more_stopwords  = ['00','000','0000','0003','0004','0004','0005'] 
more_stopwords += stopwords.words('english')
more_stopwords += stopwords.words('japanese') 
more_stopwords += stopwords.words('chinese')
more_stopwords += stopwords.words('arabic')
more_stopwords += stopwords.words('korean')
more_stopwords += stopwords.words('russian')


# In[210]:

#token_pattern = r"\b[a-z]*\b"
token_pattern = r"*"

col_desc = appdf.Description
df_desc = pd.DataFrame(col_desc, columns=['Description']).iloc[:]


stem_vectorizer = StemmedCountVectorizer(min_df =min_df,
                                         max_df= max_df,
                                         analyzer='word',
                                         stop_words =more_stopwords
                                         )
newfeature, desc_fmatrix, desc_vectorizer = vectorize_column2(df_desc, 'Description', stem_vectorizer, n_samples=100)


# In[221]:

stat_vectorized_matrix(desc_fmatrix.toarray(), desc_vectorizer)
newfeature.tail(5)


# In[222]:

tfidf_transformer  = TfidfTransformer(norm="l2").fit(desc_fmatrix)
desc_nd_array_x = tfidf_transformer.transform(desc_fmatrix, copy=True)

stat_vectorized_matrix(desc_nd_array_x.todense(),tfidf_transformer)

print desc_nd_array_x.todense()


#  In an average, one cell of a description column generates 57 features. In this way, there are 57*n_samples features geneated after vectorization.

# #### Analysis of Name Field
#     suggest some of the price for higer number of sale
#     w1: parameterized loudness of words in context

# In[ ]:

col_name = appdf.Name
#print col_name[col_name.str.contains('000')]

#stem_vectorizer = StemmedCountVectorizer(min_df =1, stop_words='english')
#newfeature, fmatrix, column_vectorizer = vectorize_column(appdf, 'Name', stem_vectorizer)
#print column_vectorizer.get_feature_names()
#newfeature.head(5)


# #### Analysis of 'Instalations'
#     It is range values.

# In[ ]:

col_name = appdf.Instalations
def separate_instalation_column(dataframe, column_name,return_data_type_as=None):
    
    col_name = appdf[column_name]
    ls = col_name.str.split('-').str.get(0).str.strip(' ').str.replace(',','') #series object
    hs = col_name.str.split('-').str.get(1).str.strip(' ').str.replace(',','') #series object
    
    if return_data_type_as is float64:
        ls = ls.astype(float).fillna(0.0)
        hs = hs.astype(float).fillna(0.0)
        return ls, hs
    else:
        return ls, hs
    
ls, hs = separate_instalation_column(appdf,'Instalations', float64)
appdf.installs_ls = ls
appdf.installs_hs = hs
print appdf.installs_ls.head(5) + appdf.installs_hs.head(5)


# # Issues:
#             Tokenization problem
#             Vectorization problem memory error
#             Plot the frequency distribution of price 
#             Do linear regression on price and plot Predicted_price-Desired_price Vs Predicted_price
#             
#             
#             http://www.cs.toronto.edu/~marlin/research/thesis/cfmlp.pdf
#             
# ### FootNotes:
#     
#         What does a rater sees when he rates an android app? == Extrinsic Features
#         What an android app inherits that influences app rating? == Intrinsic Features
# 
# 
#         Vectors to predict: 1. 5-star count, 4-star count, 3-star-count, 2-star count, 1-star count.
#         Because, average app-rating depends upon the values of these values. Also on current rating of the app.
# 

# In[ ]:



