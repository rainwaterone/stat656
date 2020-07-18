#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: EJones
"""

import sys, math, operator
import pandas as pd
import numpy  as np
from AdvancedAnalytics.Text import text_analysis, text_plot
# Packages from Sci-Learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

import matplotlib.pyplot as plt
from wordcloud   import WordCloud, STOPWORDS
from collections import Counter
from PIL         import Image

df = pd.read_excel("hotels.xlsx")
print("Hotels file contains", df.shape[0], "reviews.")

# Setup program constants and reviews
n_reviews  = len(df['Review'])
s_words    = 'english'
ngram = (1,2)
reviews = np.array(df['Review'])

# Constants
n_topics        =  9   # number of topics to classify into
max_iter        = 10   # maximum number of iterations
learning_offset = 10.      # default is 10
learning_method = 'online' # alternative is 'batch' for large files

# Create Word Frequency by Review Matrix using Custom Analyzer
# max_df is a stop limit for terms that have more than this 
# proportion of documents with the term (max_df - don't ignore any terms)
ta = text_analysis(synonyms=None, stop_words=["quot", "amp"], pos=True, stem=True )
# max_df, min_df - drops top and bottom percentage of words
# binary - True creates a binary version of the matrix
cv = CountVectorizer(max_df=0.98, min_df=0.02, max_features=None,
                     binary=False, analyzer=ta.analyzer)
tf    = cv.fit_transform(reviews)
terms = cv.get_feature_names()
print('{:.<22s}{:>6d}'.format("Number of Reviews", len(reviews)))
print('{:.<22s}{:>6d}'.format("Number of Terms",   len(terms)))
print(cv)

# Show Word Cloud based on TFIDF weighting
tfidf = True
if tfidf == True:
    # Construct the TF/IDF matrix from the data
    print("\nConducting Term/Frequency Matrix using TF-IDF")
    # Default for norm is 'l2', use norm=None to supress
    tfidf_vect = TfidfTransformer(norm=None, use_idf=True)
    # tf matrix is (n_reviews)x(m_features)
    tf = tfidf_vect.fit_transform(tf) 
    
# Turn the Term/Frequency matrix into a dictionary
td = text_plot.term_dic(tf, terms, scores=None)
# Display the top 20 terms
k  = Counter(td)
top_terms = k.most_common(20)
if type(top_terms[0][1]) == np.float64:
    for t in top_terms:
        print("{:10s}{:>8.2f}".format(t[0], t[1]))
else:
    for t in top_terms:
        print("{:10s}{:>8d}".format(t[0], t[1]))

text_plot.word_cloud_dic(td, mask=None, random=12345, bg_color="blue", 
                         max_words=30, size=(400,200))

uv = LatentDirichletAllocation(n_components=n_topics)
U  = uv.fit_transform(tf)
text_analysis.display_topics(uv, terms, n_terms=15, word_cloud=True)
        