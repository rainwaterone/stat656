"""
STAT 656 HW-10
@author:Lee Rainwater
@heavy_lifting_by: Dr. Edward Jones
@date: 2020-07-29

"""

import pandas as pd
# Classes provided from AdvancedAnalytics ver 1.25
from AdvancedAnalytics.Text          import text_analysis
from AdvancedAnalytics.Text          import sentiment_analysis
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from AdvancedAnalytics.Text          import text_plot

def heading(headerstring):
    """
    Centers headerstring on the page. For formatting to stdout
    Parameters
    ----------
    headerstring : string
    String that you wish to center.
    Returns
    -------
    Returns: None.
    """
    tw = 70 # text width
    lead = int(tw/2)-(int(len(headerstring)/2))-1
    tail = tw-lead-len(headerstring)-2
    print('\n' + ('*'*tw))
    print(('*'*lead) + ' ' + headerstring + ' ' + ('*'*tail))
    print(('*'*tw))
    return

heading("READING DATA SOURCE...")

# Set Pandas Columns Width for Excel Columns
pd.set_option('max_colwidth', 32000)
df = pd.read_excel("hotels.xlsx")

text_col  = 'Review' #Identify the Data Frame Text Target Column Name
# Check if any text was truncated
pd_width    = pd.get_option('max_colwidth')
maxsize     = df[text_col].map(len).max() # Maps text_col onto len() and finds max()
n_truncated = (df[text_col].map(len) > pd_width).sum()
print("\nTEXT LENGTH:")
print("{:<17s}{:>6d}".format("   Max. Accepted", pd_width))
print("{:<17s}{:>6d}".format("   Max. Observed", maxsize))
print("{:<17s}{:>6d}".format("   Truncated", n_truncated))

# Initialize TextAnalytics and Sentiment Analysis.  
ta = text_analysis(synonyms=None, stop_words=None, pos=False, stem=False)
# n_terms=2 only displays text containing 2 or more sentiment words for
# the list of the highest and lowest sentiment strings
sa = sentiment_analysis(n_terms=2)

heading("CREATING TOKEN COUNT MATRIX...")
# Create Word Frequency by Review Matrix using Custom Sentiment 
cv = CountVectorizer(max_df=1.0, min_df=1, max_features=None, \
                     ngram_range=(1,2), analyzer=sa.analyzer, \
                     vocabulary=sa.sentiment_word_dic)
stf        = cv.fit_transform(df[text_col]) # Return document-term matrix
sterms     = cv.get_feature_names() # Map feature indices to feature names
heading("CALCULATE AND STORE SENTIMENT SCORES...")
# Calculate and Store Sentiment Scores into DataFrame "s_score"
s_score    = sa.scores(stf, sterms)
n_reviews  = s_score.shape[0]
n_sterms   = s_score['n_words'].sum()
max_length = df['Review'].apply(len).max()
if n_sterms == 0 or n_reviews == 0:
    print("No sentiment terms found.")

p = s_score['n_words'].sum() / n_reviews
print('{:-<24s}{:>6d}'.format("\nMaximum Text Length", max_length))
print('{:-<23s}{:>6d}'.format("Total Reviews", n_reviews))
print('{:-<23s}{:>6d}'.format("Total Sentiment Terms", n_sterms))
print('{:-<23s}{:>6.2f}'.format("Avg. Sentiment Terms", p))

# s_score['sentiment'] = s_score['sentiment'].map("{:,.2f}".format)
df = df.join(s_score)
print("\n", df[['hotel', 'sentiment', 'n_words']], "\n")

print(df.groupby(['hotel']).mean())

heading("GENERATING TOTAL WORD CLOUD FOR CORPUS...")
tcv = CountVectorizer(max_df=1.0, min_df=1, max_features=None, \
                      ngram_range=(1,2), analyzer=ta.analyzer)
tf  = tcv.fit_transform(df[text_col])
terms   = tcv.get_feature_names()
td  = text_plot.term_dic(tf, terms)
text_plot.word_cloud_dic(td, max_words=200)

heading("GENERATING SENTIMENT WORD CLOUD FOR CORPUS...")
corpus_sentiment = {}
n_sw = 0
for i in range(n_reviews):
    # Iterate over the terms with nonzero scores."stf" is a sparse matrix
    term_list = stf[i].nonzero()[1]
    if len(term_list)>0:
        for t in np.nditer(term_list):
            score = sa.sentiment_dic.get(sterms[t])
            if score != None:
                n_sw += stf[i,t]
                current_count = corpus_sentiment.get(sterms[t])
                if current_count == None:
                    corpus_sentiment[sterms[t]] = stf[i,t]
                else:
                    corpus_sentiment[sterms[t]] += stf[i,t]

# Word cloud for the Sentiment Words found in the Corpus
text_plot.word_cloud_dic(corpus_sentiment, max_words=200)
n_usw = len(corpus_sentiment)
print("\nSENTIMENT TERMS")
print("------------------")
print("{:.<10s}{:>8d}".format("Unique",n_usw))
print("{:.<10s}{:>8d}".format("Total", n_sw ))
print("------------------")


heading("GENERATING TOTAL WORD CLOUD FOR BELLAGIO...")
tcv = CountVectorizer(max_df=1.0, min_df=1, max_features=None, \
                      ngram_range=(1,2), analyzer=ta.analyzer)
tf  = tcv.fit_transform(df[df['hotel']=='Bellagio'][text_col])
terms   = tcv.get_feature_names()
td  = text_plot.term_dic(tf, terms)
text_plot.word_cloud_dic(td, max_words=200)

heading("GENERATING SENTIMENT WORD CLOUD FOR BELLAGIO...")

bcv = CountVectorizer(max_df=1.0, min_df=1, max_features=None, \
                     ngram_range=(1,2), analyzer=sa.analyzer, \
                     vocabulary=sa.sentiment_word_dic)
bstf        = bcv.fit_transform(df[df['hotel']=='Bellagio'][text_col]) # Return document-term matrix
bsterms     = bcv.get_feature_names() # Map feature indices to feature names
heading("CALCULATE AND STORE SENTIMENT SCORES FOR BELLAGIO...")
# Calculate and Store Sentiment Scores into DataFrame "s_score"
bs_score    = sa.scores(bstf, bsterms)
bn_reviews  = bs_score.shape[0]
bn_sterms   = bs_score['n_words'].sum()
max_length = df['Review'].apply(len).max()
if bn_sterms == 0 or bn_reviews == 0:
    print("No sentiment terms found.")


corpus_sentiment = {}
n_sw = 0
for i in range(bn_reviews):
    # Iterate over the terms with nonzero scores."stf" is a sparse matrix
    term_list = bstf[i].nonzero()[1]
    if len(term_list)>0:
        for t in np.nditer(term_list):
            score = sa.sentiment_dic.get(bsterms[t])
            if score != None:
                n_sw += bstf[i,t]
                current_count = corpus_sentiment.get(bsterms[t])
                if current_count == None:
                    corpus_sentiment[bsterms[t]] = bstf[i,t]
                else:
                    corpus_sentiment[bsterms[t]] += bstf[i,t]

# Word cloud for the Sentiment Words found in the Corpus
text_plot.word_cloud_dic(corpus_sentiment, max_words=200)
n_usw = len(corpus_sentiment)
print("\nBELLAGIO SENTIMENT TERMS")
print("------------------")
print("{:.<10s}{:>8d}".format("Unique",n_usw))
print("{:.<10s}{:>8d}".format("Total", n_sw ))
print("------------------")

