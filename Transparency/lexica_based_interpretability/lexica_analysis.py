#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:26:09 2020

@author: jbellwoar
"""

import os
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import spatial
from adjustText import adjust_text
from matplotlib import pyplot as plt
from gensim.models import KeyedVectors

en_model = KeyedVectors.load_word2vec_format(os.path.join(Path.home(),'Downloads','wiki-news-300d-1M.vec'))

#%% Load Lexica and CSVs with attention words

# LIWC Lexica
enliwc = pd.read_csv('./liwc2015_en.csv')

# Other lexica
positive_words_df = pd.read_csv('positive-words.txt', encoding = "ISO-8859-1", header=None)
positive_words = positive_words_df[0].tolist()
negative_words_df = pd.read_csv('negative-words.txt', encoding = "ISO-8859-1", header=None) 
negative_words = negative_words_df[0].tolist()

#%% Main Functions 
def getWordVec(word):
    UNK = False
    if word in en_model.vocab:
        vec = en_model[word]
    else:
        vec = np.zeros(300)
        UNK = True
    return vec, UNK
    
def getMeanVec(wordlist):
    wordvec = []
    for word in wordlist:
        vec, UNK = getWordVec(word)
        if not UNK:
            wordvec.append(vec)
    meanvec = np.mean(wordvec, axis = 0)
    return meanvec

def getMeanVecLIWC(cate_name):
    wordlist = list(enliwc[enliwc.category==cate_name].term)
    return getMeanVec(wordlist)    

def getCosSim(vec1, vec2):
    return 1. - spatial.distance.cosine(vec1, vec2)

from numpy import linalg as LA
def getL2Norm(vec1, vec2):
    return LA.norm(vec1 - vec2)

def getCoor(wordlist): 
    coor = []
    text = []
    for word in wordlist:
        vec, UNK = getWordVec(word)
        if not UNK:
            #coor.append(getCosSim(enPosMean,vec)-getCosSim(enNegMean,vec))
            """TRYING L2 Norm instead of cosine similarity"""
            coor.append( -(getL2Norm(enPosMean,vec)-getL2Norm(enNegMean,vec)))
            text.append(word)
            """ 
            Should we use cosine similarity or L2 Norm for distance since that is what 
            nearest neighbors uses. FastText uses nearest neightbors to group similar words,
            therefore we should measure our similarity using this euclidean disncae as well.
            """ 
    return coor, text

def getCoor_v2_L2(word): 
    coor = np.nan
    vec, UNK = getWordVec(word)
    if not UNK:
        """TRYING L2 Norm instead of cosine similarity"""
        coor = -(getL2Norm(enPosMean,vec)-getL2Norm(enNegMean,vec))
    return coor

def getCoor_v2_CosSim(word): 
    coor = np.nan
    vec, UNK = getWordVec(word)
    if not UNK:
        coor = (getCosSim(enPosMean,vec)-getCosSim(enNegMean,vec))
    return coor

def project(wordlist):
    coor, text = getCoor(wordlist)
    df = pd.DataFrame({'coor': coor}) # Coor is the data 
    fig,ax = plt.subplots(figsize=(5,5),dpi=100)
    ax = sns.scatterplot(data=df, x="coor", y="coor", color = 'red')
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    texts = [ax.text(coor[i], coor[i], text[i]) for i in range(len(coor))]
    adjust_text(texts)
    ax.set_xlabel('')    
    ax.set_ylabel('')
    max_l2 = getL2Norm(enPosMean,enNegMean) # Theoretical Max Distance based on our calculation
    ax.set_xlim(-max_l2, max_l2)
    ax.set_ylim(-max_l2, max_l2)   
    #ax.set_xlim(-0.15, 0.15)
    #ax.set_ylim(-0.15, 0.15)
    print(len(coor))
    return fig

def word_hist_2df_L2(df_pos,df_neg,words_analyzed=50):    
    df_pos = df_pos[:][0:words_analyzed]
    df_neg = df_neg[:][0:words_analyzed]
    df_pos['Category'] = 'Positive'
    df_neg['Category'] = 'Negative'
    df_all = df_pos.append(df_neg,ignore_index=True)
    df_all['coor']=0
    for index, row in df_all.iterrows():
        coor = getCoor_v2_L2(row['word'])
        df_all.loc[index, 'coor']= coor
    fig,ax = plt.subplots(figsize=(5,5),dpi=100)
    ax = sns.histplot(data=df_all, x="coor", hue="Category", 
                      stat='probability', binwidth=0.02, element="step")
    max_l2 = getL2Norm(enPosMean,enNegMean) # Theoretical Max Distance based on our calculation
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(0, 0.15)
    ax.set_xlabel('Magnitude')
    # Also get stats as well
    pos_mean = np.mean(df_all['coor'][df_all['Category']=='Positive'])
    neg_mean = np.mean(df_all['coor'][df_all['Category']=='Negative'])
    all_std = np.std(df_all['coor'][:])
    return fig, pos_mean, neg_mean, all_std

def word_hist_2df_CosSim(df_pos,df_neg,words_analyzed=50):    
    df_pos = df_pos[:][0:words_analyzed]
    df_neg = df_neg[:][0:words_analyzed]
    df_pos['Category'] = 'Positive'
    df_neg['Category'] = 'Negative'
    df_all = df_pos.append(df_neg,ignore_index=True)
    df_all['coor']=0
    for index, row in df_all.iterrows():
        coor = getCoor_v2_CosSim(row['word'])
        df_all.loc[index, 'coor']= coor
    fig,ax = plt.subplots(figsize=(5,5),dpi=100)
    ax = sns.histplot(data=df_all, x="coor", hue="Category", 
                      stat='probability', binwidth=0.02, element="step")
    #max_l2 = getL2Norm(enPosMean,enNegMean) # Theoretical Max Distance based on our calculation
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(0, 0.15)
    ax.set_xlabel('Magnitude')
    # Also get stats as well
    pos_mean = np.mean(df_all['coor'][df_all['Category']=='Positive'])
    neg_mean = np.mean(df_all['coor'][df_all['Category']=='Negative'])
    all_std = np.std(df_all['coor'][:])
    return fig, pos_mean, neg_mean, all_std

#%% Get the Mean Lexica from LIWC and Other Lexicon

# LIWC Dictionary    
enPosMean = getMeanVecLIWC('POSEMO')
enNegMean = getMeanVecLIWC('NEGEMO')

# Other Lexicon
enPosMean_lex = getMeanVec(positive_words) 
enNegMean_lex = getMeanVec(negative_words)  

# Find similarity between LIWC Dict and New Lexicon
print('Cosine Similarity Between:')
print('LIWC Positive and New Lexicon Positive Vectors',getCosSim(enPosMean,enPosMean_lex))
print('LIWC Negative and New Lexicon Negative Vectors',getCosSim(enNegMean,enNegMean_lex))
print('LIWC Positive and LIWC Negative Vectors',getCosSim(enPosMean,enNegMean))
print('New Lexicon Positive Vectors and New Lexicon Negative Vectors',getCosSim(enPosMean_lex,enNegMean_lex))

# L2 Distance between Vectors
print('L2 Distance Between:')
print('LIWC Positive and New Lexicon Positive Vectors',getL2Norm(enPosMean,enPosMean_lex))
print('LIWC Negative and New Lexicon Negative Vectors',getL2Norm(enNegMean,enNegMean_lex))
print('LIWC Positive and LIWC Negative Vectors',getL2Norm(enPosMean,enNegMean))
print('New Lexicon Positive Vectors and New Lexicon Negative Vectors',getL2Norm(enPosMean_lex,enNegMean_lex))


"""
Advanced reader: measure of similarity
In order to find nearest neighbors, we need to compute a similarity score between words. 
Our words are represented by continuous word vectors and we can thus apply simple 
similarities to them. In particular we use the cosine of the angles between two vectors. 
This similarity is computed for all words in the vocabulary, and the 10 most similar words 
are shown. Of course, if the word appears in the vocabulary, it will appear on top, with a 
similarity of 1. - From FastText's website

Therefore it seems like we can use either nn given by L2 norm or the cosine similarity
"""

#%% Combine the lists 

def merge_pos_and_neg(df_pos, df_neg): 
    df_merged = df_pos.merge(df_neg,how='inner',on='word')
    df_merged['value'] = (df_merged['value_x']-df_merged['value_y'])
    df_merged.sort_values(by=['value'], ascending=False, inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    return df_merged

def get_pos_and_neg_subtracted(df_merged):
    df_pos_subtracted = df_merged[['word','value']][0:1000]
    df_merged.sort_values(by=['value'],ascending=True,inplace=True)
    df_neg_subtracted = df_merged[['word','value']][0:1000] # keep only word and value 
    return df_pos_subtracted, df_neg_subtracted


#%% Plot for IMDB
    
# Load IMDB
imdb_vanillalstm_pos_word_df = pd.read_csv('imdb_vanillalstm_pos_word.csv')
imdb_vanillalstm_neg_word_df = pd.read_csv('imdb_vanillalstm_neg_word.csv')
imdb_diversitylstm_pos_word_df = pd.read_csv('imdb_diversitylstm_pos_word.csv')
imdb_diversitylstm_neg_word_df = pd.read_csv('imdb_diversitylstm_neg_word.csv')

# Not Subtracted Lists
print('IMDB Measures: pos_mean, neg_mean, all_std')
fig, pos_mean, neg_mean, all_std = word_hist_2df_L2(imdb_vanillalstm_pos_word_df,imdb_vanillalstm_neg_word_df)
fig.suptitle('IMDB Vanilla L2'), print('Vanilla L2',pos_mean, neg_mean, all_std)
fig, pos_mean, neg_mean, all_std = word_hist_2df_L2(imdb_diversitylstm_pos_word_df,imdb_diversitylstm_neg_word_df)
fig.suptitle('IMDB Diversity L2'),print('Diversity L2',pos_mean, neg_mean, all_std)
fig, pos_mean, neg_mean, all_std = word_hist_2df_CosSim(imdb_vanillalstm_pos_word_df,imdb_vanillalstm_neg_word_df)
fig.suptitle('IMDB Vanilla Cosine Similarity'), print('Vanilla Cosine Similarity',pos_mean, neg_mean, all_std)
fig, pos_mean, neg_mean, all_std = word_hist_2df_CosSim(imdb_diversitylstm_pos_word_df,imdb_diversitylstm_neg_word_df)
fig.suptitle('IMDB Diversity Cosine Similarity'),print('Diversity Cosine Similarity',pos_mean, neg_mean, all_std)


# Subtracted List
df_merged = merge_pos_and_neg(imdb_vanillalstm_pos_word_df, imdb_vanillalstm_neg_word_df)
df_pos_subtracted, df_neg_subtracted = get_pos_and_neg_subtracted(df_merged)
fig, pos_mean, neg_mean, all_std = word_hist_2df_L2(df_pos_subtracted,df_neg_subtracted,100)
fig.suptitle('IMDB Vanilla L2 (Subtracted Lists)'), print('Vanilla L2',pos_mean, neg_mean, all_std)

df_merged = merge_pos_and_neg(imdb_diversitylstm_pos_word_df, imdb_diversitylstm_neg_word_df)
df_pos_subtracted, df_neg_subtracted = get_pos_and_neg_subtracted(df_merged)
fig, pos_mean, neg_mean, all_std = word_hist_2df_L2(df_pos_subtracted,df_neg_subtracted,100)
fig.suptitle('IMDB Diversity L2 (Subtracted Lists)'),print('Diversity L2',pos_mean, neg_mean, all_std)

df_merged = merge_pos_and_neg(imdb_vanillalstm_pos_word_df, imdb_vanillalstm_neg_word_df)
df_pos_subtracted, df_neg_subtracted = get_pos_and_neg_subtracted(df_merged)
fig, pos_mean, neg_mean, all_std = word_hist_2df_CosSim(df_pos_subtracted,df_neg_subtracted,100)
fig.suptitle('IMDB Vanilla Cosine Similarity (Subtracted Lists)'), print('Vanilla Cosine Similarity',pos_mean, neg_mean, all_std)

df_merged = merge_pos_and_neg(imdb_diversitylstm_pos_word_df, imdb_diversitylstm_neg_word_df)
df_pos_subtracted, df_neg_subtracted = get_pos_and_neg_subtracted(df_merged)
fig, pos_mean, neg_mean, all_std = word_hist_2df_CosSim(df_pos_subtracted,df_neg_subtracted,100)
fig.suptitle('IMDB Diversity Cosine Similarity (Subtracted Lists)'),print('Diversity Cosine Similarity',pos_mean, neg_mean, all_std)


""" SHOULD WE STACK THESE HISTOGRAMS??? (makes trend easier to see) """

#%% Plot Yelp

# Load Yelp
Yelp_vanillalstm_pos_word_df = pd.read_csv('Yelp_vanillalstm_pos_word.csv')
Yelp_vanillalstm_neg_word_df = pd.read_csv('Yelp_vanillalstm_neg_word.csv')
Yelp_diversitylstm_pos_word_df = pd.read_csv('Yelp_diversitylstm_pos_word.csv')
Yelp_diversitylstm_neg_word_df = pd.read_csv('Yelp_diversitylstm_neg_word.csv')

# Not Subtracted List
print('Measures: pos_mean, neg_mean, all_std')
fig, pos_mean, neg_mean, all_std = word_hist_2df_L2(Yelp_vanillalstm_pos_word_df,Yelp_vanillalstm_neg_word_df)
fig.suptitle('Yelp Vanilla L2'), print('Vanilla L2',pos_mean, neg_mean, all_std)
fig, pos_mean, neg_mean, all_std = word_hist_2df_L2(Yelp_diversitylstm_pos_word_df,Yelp_diversitylstm_neg_word_df)
fig.suptitle('Yelp Diversity L2'),print('Diversity L2',pos_mean, neg_mean, all_std)
fig, pos_mean, neg_mean, all_std = word_hist_2df_CosSim(Yelp_vanillalstm_pos_word_df,Yelp_vanillalstm_neg_word_df)
fig.suptitle('Yelp Vanilla Cosine Similarity'), print('Vanilla Cosine Similarity',pos_mean, neg_mean, all_std)
fig, pos_mean, neg_mean, all_std = word_hist_2df_CosSim(Yelp_diversitylstm_pos_word_df,Yelp_diversitylstm_neg_word_df)
fig.suptitle('Yelp Diversity Cosine Similarity'),print('Diversity Cosine Similarity',pos_mean, neg_mean, all_std)
# These look good

# Subtracted List
print('Yelp Measures: pos_mean, neg_mean, all_std')
df_merged = merge_pos_and_neg(Yelp_vanillalstm_pos_word_df, Yelp_vanillalstm_neg_word_df)
df_pos_subtracted, df_neg_subtracted = get_pos_and_neg_subtracted(df_merged)
fig, pos_mean, neg_mean, all_std = word_hist_2df_L2(df_pos_subtracted,df_neg_subtracted,100)
fig.suptitle('Yelp Vanilla L2 (Subtracted Lists)'), print('Vanilla L2',pos_mean, neg_mean, all_std)

df_merged = merge_pos_and_neg(Yelp_diversitylstm_pos_word_df, Yelp_diversitylstm_neg_word_df)
df_pos_subtracted, df_neg_subtracted = get_pos_and_neg_subtracted(df_merged)
fig, pos_mean, neg_mean, all_std = word_hist_2df_L2(df_pos_subtracted,df_neg_subtracted,100)
fig.suptitle('Yelp Diversity L2 (Subtracted Lists)'),print('Diversity L2',pos_mean, neg_mean, all_std)

df_merged = merge_pos_and_neg(Yelp_vanillalstm_pos_word_df, Yelp_vanillalstm_neg_word_df)
df_pos_subtracted, df_neg_subtracted = get_pos_and_neg_subtracted(df_merged)
fig, pos_mean, neg_mean, all_std = word_hist_2df_CosSim(df_pos_subtracted,df_neg_subtracted,100)
fig.suptitle('Yelp Vanilla Cosine Similarity (Subtracted Lists)'), print('Vanilla Cosine Similarity',pos_mean, neg_mean, all_std)

df_merged = merge_pos_and_neg(Yelp_diversitylstm_pos_word_df, Yelp_diversitylstm_neg_word_df)
df_pos_subtracted, df_neg_subtracted = get_pos_and_neg_subtracted(df_merged)
fig, pos_mean, neg_mean, all_std = word_hist_2df_CosSim(df_pos_subtracted,df_neg_subtracted,100)
fig.suptitle('Yelp Diversity Cosine Similarity (Subtracted Lists)'),print('Diversity Cosine Similarity',pos_mean, neg_mean, all_std)


# Checking the vanilla and diversity against each other
df_merged_vanilla = merge_pos_and_neg(Yelp_vanillalstm_pos_word_df, Yelp_vanillalstm_neg_word_df)
df_pos_subtracted_vanilla, df_neg_subtracted_vanilla = get_pos_and_neg_subtracted(df_merged_vanilla)

df_merged_diversity = merge_pos_and_neg(Yelp_diversitylstm_pos_word_df, Yelp_diversitylstm_neg_word_df)
df_pos_subtracted_diversity, df_neg_subtracted_diversity = get_pos_and_neg_subtracted(df_merged_diversity)

#%% Let's check the overall distribtution of all the weights values

"""
plt.hist(Yelp_vanillalstm_pos_word_df['value'])
plt.hist(Yelp_vanillalstm_neg_word_df['value'])
plt.hist(Yelp_diversitylstm_pos_word_df['value'])
plt.hist(Yelp_diversitylstm_neg_word_df['value'])
"""

#%% SST 

#Load SST
sst_vanillalstm_pos_word_df = pd.read_csv('sst_vanillalstm_pos_word.csv')
sst_vanillalstm_neg_word_df = pd.read_csv('sst_vanillalstm_neg_word.csv')
sst_diversitylstm_pos_word_df = pd.read_csv('sst_diversitylstm_pos_word.csv')
sst_diversitylstm_neg_word_df = pd.read_csv('sst_diversitylstm_neg_word.csv')

# Not Subtracted List
print('SST Measures: pos_mean, neg_mean, all_std')
fig, pos_mean, neg_mean, all_std = word_hist_2df_L2(sst_vanillalstm_pos_word_df,sst_vanillalstm_neg_word_df)
fig.suptitle('SST Vanilla L2'), print('Vanilla L2',pos_mean, neg_mean, all_std)
fig, pos_mean, neg_mean, all_std = word_hist_2df_L2(sst_diversitylstm_pos_word_df,sst_diversitylstm_neg_word_df)
fig.suptitle('SST Diversity L2'),print('Diversity L2',pos_mean, neg_mean, all_std)
fig, pos_mean, neg_mean, all_std = word_hist_2df_CosSim(sst_vanillalstm_pos_word_df,sst_vanillalstm_neg_word_df)
fig.suptitle('SST Vanilla Cosine Similarity'), print('Vanilla Cosine Similarity',pos_mean, neg_mean, all_std)
fig, pos_mean, neg_mean, all_std = word_hist_2df_CosSim(sst_diversitylstm_pos_word_df,sst_diversitylstm_neg_word_df)
fig.suptitle('SST Diversity Cosine Similarity'),print('Diversity Cosine Similarity',pos_mean, neg_mean, all_std)


# Subtracted List
df_merged = merge_pos_and_neg(sst_vanillalstm_pos_word_df, sst_vanillalstm_neg_word_df)
df_pos_subtracted, df_neg_subtracted = get_pos_and_neg_subtracted(df_merged)
fig, pos_mean, neg_mean, all_std = word_hist_2df_L2(df_pos_subtracted,df_neg_subtracted,100)
fig.suptitle('SST Vanilla L2 (Subtracted Lists)'), print('Vanilla L2',pos_mean, neg_mean, all_std)

df_merged = merge_pos_and_neg(sst_diversitylstm_pos_word_df, sst_diversitylstm_neg_word_df)
df_pos_subtracted, df_neg_subtracted = get_pos_and_neg_subtracted(df_merged)
fig, pos_mean, neg_mean, all_std = word_hist_2df_L2(df_pos_subtracted,df_neg_subtracted,100)
fig.suptitle('SST Diversity L2 (Subtracted Lists)'),print('Diversity L2',pos_mean, neg_mean, all_std)

df_merged = merge_pos_and_neg(sst_vanillalstm_pos_word_df, sst_vanillalstm_neg_word_df)
df_pos_subtracted, df_neg_subtracted = get_pos_and_neg_subtracted(df_merged)
fig, pos_mean, neg_mean, all_std = word_hist_2df_CosSim(df_pos_subtracted,df_neg_subtracted,100)
fig.suptitle('SST Vanilla Cosine Similarity (Subtracted Lists)'), print('Vanilla Cosine Similarity',pos_mean, neg_mean, all_std)

df_merged = merge_pos_and_neg(sst_diversitylstm_pos_word_df, sst_diversitylstm_neg_word_df)
df_pos_subtracted, df_neg_subtracted = get_pos_and_neg_subtracted(df_merged)
fig, pos_mean, neg_mean, all_std = word_hist_2df_CosSim(df_pos_subtracted,df_neg_subtracted,100)
fig.suptitle('SST Diversity Cosine Similarity (Subtracted Lists)'),print('Diversity Cosine Similarity',pos_mean, neg_mean, all_std)

#%% Correlation between pre and post merge
from scipy import signal
#scipy.signal.correlate

# Checking the vanilla and diversity against each other
Yelp_vanillalstm_pos_word_df = pd.read_csv('Yelp_vanillalstm_pos_word.csv')
Yelp_vanillalstm_neg_word_df = pd.read_csv('Yelp_vanillalstm_neg_word.csv')
Yelp_diversitylstm_pos_word_df = pd.read_csv('Yelp_diversitylstm_pos_word.csv')
Yelp_diversitylstm_neg_word_df = pd.read_csv('Yelp_diversitylstm_neg_word.csv')


df_merged_diversity = merge_pos_and_neg(Yelp_diversitylstm_pos_word_df, Yelp_diversitylstm_neg_word_df)

for index, row in df_merged_diversity.iterrows():
        coor = getCoor_v2_CosSim(row['word'])
        df_merged_diversity.loc[index, 'coor']= coor

#%% 

df_merged_diversity.sort_values(by=['value'],ascending=False,inplace=True)
df_merged_diversity_mod = df_merged_diversity.dropna(inplace=False)

#from sklearn.metrics import r2_score
#r_score = r2_score(df_merged_diversity_mod['value'][0:10],df_merged_diversity_mod['coor'][0:10])
#print(r_score)

plt.scatter(df_merged_diversity_mod['value'],df_merged_diversity_mod['coor'])
plt.xlim([-10,10])
plt.axhline(0,color='r')
plt.axvline(0,color='r')
plt.ylabel('Magnitude'),plt.xlabel('Subtraction Value')

