# -*- coding: utf-8 -*-
"""
Calculating distances between Census tweets and respondant quote
"""


#### Packages
import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer # word-doc matrix
from scipy.sparse import coo_matrix # sparse matrices
from scipy.sparse import csr_matrix # sparse row matrix

# for pre-processing
import string
import nltk
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords

import math # log for word distances

# LDA
from sklearn.decomposition import LatentDirichletAllocation

# LSA
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\b\w{3,}\b')
from sklearn.decomposition import TruncatedSVD


#############
### Functions
#############

# pre-processing of tweets
def preProcessingFcn(tweet, removeWords=list(), stem=True, removeURL=True, removeStopwords=True, 
    removeNumbers=False, removePunctuation=True):
    """
    Cleans tweets by removing words, stemming, etc.
    """
    ps = PorterStemmer()
    tweet = tweet.lower()
    tweet = re.sub(r"\\n", " ", tweet)
    tweet = re.sub(r"&amp", " ", tweet)
    if removeURL==True:
        tweet = re.sub(r"http\S+", " ", tweet)
    if removeNumbers==True:
        tweet=  ''.join(i for i in tweet if not i.isdigit())
    if removePunctuation==True:
        for punct in string.punctuation:
            tweet = tweet.replace(punct, ' ')
    if removeStopwords==True:
        tweet = ' '.join([word for word in tweet.split() if word not in stopwords.words('english')])
    if len(removeWords)>0:
        tweet = ' '.join([word for word in tweet.split() if word not in removeWords])
    if stem==True:
        tweet = ' '.join([ps.stem(word) for word in tweet.split()])
    return tweet

# create word-document matrix
def make_wordDocMatrix(cleanedTweets, minMentions = 5):
    vectorizer = CountVectorizer(strip_accents='unicode', min_df=minMentions, binary=True)
    docWordMatrix = vectorizer.fit_transform(cleanedTweets)
    colWords = vectorizer.get_feature_names()
    output = dict()
    output['docWordMatrix'] = docWordMatrix
    output['words'] = colWords
    return output

# create word-distance matrix
def get_wordDists(w, colWords, quote_cleaned):
    # word distances
    v = w.shape[1]
    n = w.shape[0]
    word_dists = []
    for word in list(set(quote_cleaned.split())):
        if word in colWords:
            # column of doc-word matrix associated with quote word
            quoteword_col = w[:,colWords.index(word)]
            quoteword_mat = quoteword_col * coo_matrix(np.ones([1, v]))
            multiplied = quoteword_mat.multiply(w)
            mult_cols = multiplied.sum(axis=0).tolist()[0]
            quote_sums = quoteword_mat.sum(axis=0).tolist()[0]
            w_sums = w.sum(axis=0).tolist()[0]
            add_cols = [quote_sums[i]+w_sums[i]-mult_cols[i] for i in range(v)]
            word_dists.append([1-mult_cols[i]/add_cols[i] if add_cols[i]>0 else 1 for i in range(v)])
            #word_dists.append([-1*math.log(mult_cols[i]/add_cols[i]+.001) if add_cols[i]>0 else -1*log(.001) for i in range(v)])
    word_dists_matrix = np.vstack(word_dists)
    # tweet distances
    tweetDists = []
    quoteWords = [colWords.index(word) for word in list(set(quote_cleaned.split())) if word in colWords]
    for i in range(n):
        iWords = list(np.where(w[i,:].toarray()[0]>0)[0])
        if len(iWords)==0:
            # if tweet i contains no words in word-doc matrix --> cannot compute distance
            tweet_dist = 1
        else:
            iquoteIntersect = list(set(iWords) & set(quoteWords))
            dtweetsij = word_dists_matrix[:, iWords]
            entries = dtweetsij.shape[0] * dtweetsij.shape[1]
            tweet_dist = np.sum(dtweetsij)/entries
            if len(iquoteIntersect)>0:
                dtweetsij_intersect = word_dists_matrix[:,iquoteIntersect]
                dtweetsij_intersect = dtweetsij_intersect[[quoteWords.index(j) for j in iquoteIntersect], :]
                tweet_dist = tweet_dist - np.sum(dtweetsij_intersect)/entries
        tweetDists.append(tweet_dist)
    return tweetDists
    

# LDA: new 'docWordMatrix' for quote
def quote_docWordMat(quote_cleaned, colWords):
    cols = []
    data = []
    for word in list(set(quote_cleaned.split())):
        if word in colWords:
            cols.append(colWords.index(word))
            data.append(quote_cleaned.split().count(word))
    new_docWordMat = csr_matrix((np.array(data), (np.repeat(0, len(cols)), np.array(cols))), 
                                shape=(1, len(colWords)))
    return new_docWordMat

def LDA_dists(quote_distribution, tweet_distributions, method='Euclidean'):
    distances = []
    for tweet_distribution in tweet_distributions:
        if method=='Euclidean':
            distances.append(np.linalg.norm(quote_distribution-tweet_distribution))
    return distances


# LSA
def train_LSA(tweets, n_components=20):
    tfidf = TfidfVectorizer(lowercase=True,
                            tokenizer=tokenizer.tokenize,
                            max_df=.2)
    tfidf_train_sparse = tfidf.fit_transform(tweets)
    tfidf_train_df = pd.DataFrame(tfidf_train_sparse.toarray(), 
                        columns=tfidf.get_feature_names())
    lsa_obj = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_obj.fit(tfidf_train_df)
    tfidf_lsa_data = lsa_obj.transform(tfidf_train_df)
    output = dict()
    output['tfidf_lsa_data'] = tfidf_lsa_data
    output['tfidf'] = tfidf
    output['lsa_obj'] = lsa_obj
    return output

def quote_LSA(quote_cleaned, tfidf, lsa_obj):
    tfidf_quote = tfidf.transform([quote_cleaned])
    lsa_quote = lsa_obj.transform(tfidf_quote)
    return lsa_quote

    
            

#############
### Testing
#############

#### Sample of already cleaned tweets
# read in sample of already cleaned tweets
os.chdir('')
allMessages = pd.read_csv('allCensus_sample.csv')
allMessages['cleaned_rt'] = [tweet[3:] if tweet[0:2]=='rt' else tweet for tweet in allMessages['cleaned']]
tweets_cleaned = allMessages['cleaned_rt']
tweets_cleaned = tweets_cleaned.drop_duplicates()
tweets = tweets_cleaned

# create word-document matrix
docWordMatrix = make_wordDocMatrix(tweets, minMentions=3)
w = docWordMatrix['docWordMatrix']
colWords = docWordMatrix['words']

# LDA
lda = LatentDirichletAllocation(n_components=20, random_state=0)
lda.fit(w)
topicDists = lda.transform(w)

# LDA
lsa = train_LSA(tweets)

# add quote
quote ='''
The people came around to my door and I felt like it was none of their business. 
What goes on in my house is totally private. The government shouldn’t care. 
I don’t think they deserve to know what’s going on in my home. 
They don’t do anything to help me, why should I answer any questions for them?
'''

quote = '''
[Census information is shared] with the entire government. With everyone in the government…
police, immigration, hospitals, everything, everything, everything. Everything is connected.
'''

quote = '''
The government has always been intrusive as it is, and it’s probably a level of intrusion. 
That’s why people are like, ‘Hold on, what you want to know what’s in my bed, 
at my house, and who’s using my toilet? You should go mind your business.
'''

quote = '''
[Latinos will not participate] out of fear…[there] is practically a hunt [for us] …
and many of us Latinos are going to be afraid to be counted because of the 
retaliation that could happen because it's like giving the government information,
 of saying, ‘Oh, there are more here.
 '''
 
quote = '''
 I think it’s a necessity. I think the immigrants need to get out of here, 
 you know what I mean. I mean, even with a green card—I don’t agree with it. 
 I just don’t.
 '''


quote_cleaned = preProcessingFcn(quote)


### My Method
# tweet distances
tweetDists = get_wordDists(w, colWords, quote_cleaned)

# get closest n tweets
n_min_values = 10
rowValues = sorted(range(len(tweetDists)), key=lambda k: tweetDists[k])[:n_min_values] 
indexValues = tweets_cleaned.index[rowValues]

list(allMessages['Message'].loc[indexValues])


### LDA
quote_w = quote_docWordMat(quote_cleaned, colWords)
quoteDist = lda.transform(quote_w)

# distances
dists = LDA_dists(quoteDist[0], topicDists)

# get closest n tweets
n_min_values = 10
rowValues = sorted(range(len(dists)), key=lambda k: dists[k])[:n_min_values]
indexValues = tweets_cleaned.index[rowValues]

list(allMessages['Message'].loc[indexValues])


### LSA
# fit LSA to quote
lsa_quote_dist = quote_LSA(quote_cleaned, lsa['tfidf'], lsa['lsa_obj'])

# get distances
dists_LSA = LDA_dists(lsa_quote_dist, lsa['tfidf_lsa_data'])

# get closest n tweets
n_min_values = 10
rowValues = sorted(range(len(dists_LSA)), key=lambda k: dists_LSA[k])[:n_min_values]
indexValues = tweets_cleaned.index[rowValues]

list(allMessages['Message'].loc[indexValues])
