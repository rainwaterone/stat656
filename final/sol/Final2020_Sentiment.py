import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
# Classes provided from AdvancedAnalytics ver 1.25
from AdvancedAnalytics.Text          import text_analysis, text_plot
from AdvancedAnalytics.Text          import sentiment_analysis
from sklearn.feature_extraction.text import CountVectorizer
from datetime import date

# Set date for filenames
today = str(date.today())
today = today[5:]
# Set Pandas Columns Width for Excel Columns
pd.set_option('max_colwidth', 32767)
Analysis = "Covid"
filename = "PICKLE/"+Analysis+"_"+today+"_df_TA_pickle.pkl"
filename = "PICKLE/Covid_08-04_df_TA_pickle.pkl"
if Analysis == "Election":
    # Read Nov2020 Election ARTICLES
    print("{:-<40s}{:->39s}".format('*', '*'))
    print("{:-<22s}    SENTIMENT ANALYSIS OF 2020     {:->22s}".format('*','*'))
    print("{:-<22s}    PRESIDENTIAL ELECTION NEWS     {:->22s}".format('*','*'))
    print("{:-<40s}{:->39s}".format('*', '*'))
    df = pd.read_pickle(filename)
else:
    # Read COVID ARTICLES
    print("{:-<40s}{:->39s}".format('*', '*'))
    print("{:-<22s}        SENTIMENT ANALYSIS  OF     {:->22s}".format('*','*'))
    print("{:-<22s}        COVID-19 NEWS ARTICLES     {:->22s}".format('*','*'))
    print("{:-<40s}{:->39s}".format('*', '*'))
    df = pd.read_pickle(filename)

text  = 'text' # Identify the DataFrame Text Column
# Check if any text was truncated
pd_width    = pd.get_option('max_colwidth')
n_articles  = df.shape[0]
maxsize     = df[text].map(len).max()
n_truncated = (df[text].map(len) >= pd_width).sum()
print("\nArticle Truncation Summary")
print("{:-<22s}{:->21s}".format('*', '*'))
print("{:<25s}{:>6d} characters".format("   Maximum Possible Size", pd_width))
print("{:<25s}{:>6d} characters".format("   Longest Article Found", maxsize))
if n_truncated>0:
    print("{:<25s}{:>6d}".format("   Articles Truncated", n_truncated))
else:
    print("{:<25s}{:>6d}".format("   Articles Truncated", 0))
print("{:<25s}{:>6d}".format("   Articles Total", n_articles))
print("{:-<22s}{:->21s}".format('*', '*'))

print("\n{:-<22s}   STARTING TEXT PARSE & FILTER    {:->22s}\n".
      format('*', '*'))
# Initialize TextAnalytics and Sentiment Analysis.  
ta = text_analysis(synonyms=None, stop_words=None, pos=False, stem=False)
# n_terms=4 only displays text containing 4 or more sentiment words for
# the list of the highest and lowest sentiment strings
sa = sentiment_analysis(n_terms=4)

# Create Word Frequency by Review Matrix using Custom Sentiment 
cv = CountVectorizer(max_df=1.0, min_df=1, max_features=None, \
                     ngram_range=(1,2), analyzer=sa.analyzer, \
                     vocabulary=sa.sentiment_word_dic)
stf        = cv.fit_transform(df[text])
sterms     = cv.get_feature_names()
# Calculate and Store Sentiment Scores into DataFrame "s_score"
s_score    = sa.scores(stf, sterms)
n_articles = s_score.shape[0]
n_sterms   = s_score['n_words'].sum()
max_length = df[text].apply(len).max()
if n_sterms == 0 or n_articles == 0:
    print("No sentiment terms found.")

# Place sentiment scores and n_words into the original data
df['sentiment']    = s_score['sentiment']
df['n_sentiment']  = s_score['n_words']
# Standardize number of sentiment words to words/1000 terms in article
df['s_words_1k'] = 1000*df['n_sentiment']/df['length']
# Store Special Words per 1,000 Article Words
df['trump_1k']   = 1000*df['trump']/df['length']
df['biden_1k']   = 1000*df['biden']/df['length']
df['fauci_1k']   = 1000*df['fauci']/df['length']

print('{:-<34s}{:>7d}'.format("\nMaximum Text Length", max_length))
print('{:-<33s}{:>7d}'.format("Total articles", n_articles))
print('{:-<33s}{:>7d}'.format("Total Sentiment Terms", n_sterms))
print('{:-<33s}{:>7.1f}'.
      format("Avg. Sentiment Terms/1k Words", df['s_words_1k'].mean()))

df_s = df.groupby('agency')['sentiment'].mean()
#df_s = df.groupby('topic')['sentiment'].mean()
#df_s = df.groupby('source')['sentiment'].mean()
plt.style.use('dark_background')
nb_colors = plt.rcParams['axes.prop_cycle']
cn = 'maroon'
cn = '#fa8174'
cp = 'tab:blue'
cp = '#8dd3c7'
colors = []
for i in range(df_s.shape[0]):
    if df_s.iloc[i] < 0:
        colors.append(cn)
    else:
        colors.append(cp)
xpos = np.arange(df_s.shape[0])
plt.figure(figsize=(10,8))
plt.style.use('dark_background')
plt.barh(xpos, df_s, align='center', color=colors,
        path_effects=[path_effects.withSimplePatchShadow(offset=(3,-3), 
                                    alpha=0.5,shadow_rgbFace='0.3')])
ax = plt.gca()
ax.set_facecolor('black')
plt.axvline(0, color='gray')
plt.grid(b=True, which='major', axis='y', ls='dotted')
plt.yticks(xpos, df_s.index, rotation='horizontal', fontsize=16)
plt.xlabel("Sentiment", fontsize=18)
plt.show()

plt.figure(figsize=(10,6))
plt.bar(xpos, df_s, align='center', color=colors,
        path_effects=[path_effects.withSimplePatchShadow(offset=(3,-3), 
                                    alpha=0.5,shadow_rgbFace='0.3')])
ax = plt.gca()
#ax.set_facecolor('gainsboro')
ax.set_facecolor('black')
plt.axhline(0, color='gray')
plt.grid(b=True, which='major', axis='x', ls='dotted')
plt.xticks(xpos, df_s.index, rotation='75')
plt.ylabel("Sentiment", fontsize=18)
plt.show()

# Word cloud for the entire the Sentiment Words found in the Corpus
corpus_sentiment = text_plot.term_dic(stf, sterms)
text_plot.word_cloud_dic(corpus_sentiment,   max_words=100)

# Word Cloud for Sources, the sentiment words
for s in df['source'].unique():
    corpus_sentiment = {}
    n_sw = 0
    print('Sentiment Cloud for ', s)
    for i in range(n_articles):
        if df['source'].iloc[i]==s:
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
    # Word cloud for the entire the Sentiment Words found in the Corpus
    text_plot.word_cloud_dic(corpus_sentiment, max_words=100)
    
# Word Cloud for Agency, the sentiment words
for a in df['agency'].unique():
    corpus_sentiment = {}
    n_sw = 0
    print('Sentiment Cloud for ', a)
    for i in range(n_articles):
        if df['agency'].iloc[i]==a:
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
    # Word cloud for the entire the Sentiment Words found in the Corpus
    text_plot.word_cloud_dic(corpus_sentiment, max_words=100)
    
filename = "PICKLE/"+Analysis+"_"+today+"_df_all_pickle.pkl"
df.to_pickle(filename)
print("")
print("{:-<40s}{:->39s}".format('*', '*'))
print("{:-<22s}   SENTIMENT ANALYSIS COMPLETED   {:->23s}".format('*', '*'))
print("{:-<40s}{:->39s}".format('*', '*'))
