import sys, math
import pandas as pd
import numpy  as np
# Classes provided for the course
from AdvancedAnalytics.Text import text_analysis
# Packages from Sci-Learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from datetime import date

def score_topics(u, scores=True, display=True):
    topics = True
    if topics==False and scores==False:
        return None
    n_reviews = u.shape[0]
    if n_reviews <= 0:
        print("Number of reviews is zero")
        sys.exit()
    n_topics  = u.shape[1]
    if n_topics <= 0:
        print("Number of topics is zero")
        sys.exit()
    doc_prob  = np.array([0.0]*n_reviews, dtype=float)
    doc_topic = np.array([0]*n_reviews,   dtype=int)
    for i in range(n_reviews):
        doc_prob[i]  = u[i].max()
        for j in range(n_topics):
            if u[i][j] == doc_prob[i]:
                doc_topic[i] = j
                continue
    z = np.asarray([doc_topic, doc_prob])
    z = z.T
    df_prob  = pd.DataFrame(doc_prob.T, columns=['prob'])
    df_topic = pd.DataFrame(doc_topic.T, columns=['topic'])
    df = df_topic.join(df_prob)

    if display==True:
        df0 = df.groupby('topic').count()
        df0 = df0.rename(columns={'prob':'N'})
        df0.index = df0.index.astype('int')
        print("\n  ***Topic Counts***\n")
        print("  Topic     N     P")
        df0['P'] = 0.0 * n_topics
        for t in (range(n_topics)):
            x = df0['N'].iloc[t]*100/n_reviews
            df0['P'].iloc[t] = x
        
        df0.plot.bar(y='P', xlim=(0, n_topics-1), fontsize=14)
        
        for t in (range(n_topics)):
            print("    {:<5d}{:>5d}{:>7.1f}%".format(t, \
                  df0['N'].iloc[t], df0['P'].iloc[t]))
        if scores==False:
            return df_topic
        else:
            return df 

# Set date for filenames
today = str(date.today())
today = today[5:]
#*****************************************************************************
# Increase column width to let pandy read large text columns
pd.set_option('max_colwidth', 32767)
Analysis = "Election"
if Analysis == "Election":
    # Read Nov2020 Election ARTICLES
    print("{:-<40s}{:->39s}".format('*', '*'))
    print("{:-<22s}       TOPIC ANALYSIS OF 2020      {:->22s}".format('*','*'))
    print("{:-<22s}     PRESIDENTIAL ELECTION NEWS    {:->22s}".format('*','*'))
    print("{:-<40s}{:->39s}".format('*', '*'))
    df = pd.read_excel("Final2020/Nov2020_Elections_8_03_all_agency.xlsx")
else:
    # Read COVID ARTICLES
    print("{:-<40s}{:->39s}".format('*', '*'))
    print("{:-<22s}           TOPIC ANALYSIS OF       {:->22s}".format('*','*'))
    print("{:-<22s}        COVID-19 NEWS ARTICLES     {:->22s}".format('*','*'))
    print("{:-<40s}{:->39s}".format('*', '*'))
    df = pd.read_excel("Final2020/Covid_8_03_all_agency.xlsx")

text        = 'text' # Dataframe columns name for the full article
n_articles  = df.shape[0]# Check to see if documents (descriptions) were truncated
pd_width    = pd.get_option('max_colwidth')
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
# Setup program constants and articles
ngram    = (1,2)
articles = np.array(df[text])

# Constants
n_topics  =  9    # Requested number of topic clusters
max_terms = 15    # Max terms to describe in topic clusters
tfidf     = True  # Use TF-IDF Weighting if True

# Create Word Frequency by Review Matrix using Custom Analyzer
# Synonym Dictionary
syn = {"covid-19": 'covid', "coronavirus": 'covid', "heath": 'health',
       "cultur": 'culture', "joe biden": 'biden', "donald trump": 'trump',
       "vice president": 'vp', "al-jazerra": 'aljazerra', 
       "schoolhouse": 'school house', "jul": 'july', "aug": 'august',
       "bringme": 'bring me'}

# Additional Stop Words
sw = ['also', 'aj', 'nr', 'xa0', 'x2019', 'een', 'het', 'de', 'thing',
      'x201c', 'x201d', 'xa0', 'apos', '1h', 'et', 'ap', 'en', 'le',
      'der', 'les', 'un', 'und', '4h', 'j3', 'al', 'da', '20pm',
      'bihar', 'juuso', 'que', 'die', 'em', 'il', 'na', 'du', 'zu',
      'se', 'au', 'para', 'je', 'dos', 'von', 'pour', '2h', 'crr', 'une',
      'ov', 'ipl', 'van', 'est', 'dans']
ta = text_analysis(synonyms=syn, stop_words=sw, pos=True, stem=True)

max_df = 1.00 # max proportion of docs/term allowed
if max_df > 1:
    max_docs = max_df
else:
    max_docs = math.floor(max_df*n_articles)
print("max_df=",max_df,"exclude terms that appear in more than", 
      max_docs, "articles")
if Analysis == "Election":
    min_df = 25 # min number of articles required for Election data
else:
    min_df = 40 # min number of articles required for Covid data
    
if min_df > 1:
    min_docs = min_df
else:
    min_docs = math.ceil(min_df*n_articles)
print("min_df=",min_df,"exclude terms that appear in less than", 
      min_docs, "articles")
cv = CountVectorizer(max_df=max_df, min_df=min_df, max_features=None,
                     binary=False,  ngram_range=ngram, analyzer=ta.analyzer)
tf      = cv.fit_transform(articles)
terms   = cv.get_feature_names()
n_terms = len(terms)
print('{:.<22s}{:>6d}'.format("Number of Articles", n_articles))
print('{:.<22s}{:>6d}'.format("Number of Terms",   n_terms))
print("\n{:-<22s}  BUILDING TERM-FREQUENCY MATRIX  {:->23s}".format('*', '*'))

#************************ TERM WEIGHTING *************************************
# Apply TFIDF weighting if tfidf is set to True
if tfidf != True:
    #****** DISPLAY MOST AND LEAST COMMON TERMS WITHOUT TDIDF WEIGHTING ******
    text_analysis.display_term_frequency(tf, terms, n_tail=25)
else:
    # Use TFIDF Weighting
    print("\nModifying Term/Frequency Matrix using TF-IDF")
    # Default for norm is 'l2', use norm=None to supress
    tfidf_vect = TfidfTransformer(norm=None, use_idf=True)
    # tf matrix is (n_articles)x(m_features)
    tf_idf = tfidf_vect.fit_transform(tf)
    #********* DISPLAY MOST AND LEAST COMMON TERMS WITH TDIDF WEIGHTING ******
    text_analysis.display_term_frequency(tf, terms, n_tail=25, tfidf=tf_idf) 
        
# Compute term/freq decomposition described by dcomp
dcomp='lda'
print("{:-<18s}       STARTING TERM-FREQUENCY MATRIX      {:->18s}".
                                     format('*', '*'))
print("{:-<18s}           DECOMPOSITION USING {:3s}         {:->18s}".
                                     format('*', dcomp.upper(), '*'))
if dcomp == 'lda':
    # LDA Analysis
    uv = LatentDirichletAllocation(n_components=n_topics, 
            max_iter=10, learning_method='online', 
            learning_offset=10,random_state=12345)
    
elif dcomp == 'svd':
    # SVD is synonymous with LSA (Latent Semantic Analysis)
    # Currently only works with TFIDF
    uv = TruncatedSVD(n_components=n_topics, algorithm='arpack',
                                    tol=0.0, random_state=12345)
   
elif dcomp == 'nmf':
    #Non-negative Matrix Factorization
    uv = NMF(n_components=n_topics, random_state=12345, 
             alpha=0.1, l1_ratio=0.5)

elif dcomp == 'kld':
    #Kullback-Liebler Non-negative Matrix Factorization
    uv = NMF(n_components=n_topics, random_state=12345, 
             alpha=0.1,l1_ratio=0.5, 
             beta_loss='kullback-leibler', solver='mu', 
             max_iter=1000)
else:
    raise RuntimeError("dcomp is not lda, svd, nmf or kld")
    sys.exit()

if tfidf:
    print("{:-<18s}  IDENTIFYING TOPIC CLUSTERS USING TFIDF   {:->18s}".
                                                           format('*', '*'))
    U    = uv.fit_transform(tf_idf)
else:
    U    = uv.fit_transform(tf)
    fmt0 = "{:-<12s} IDENTIFYING TOPIC CLUSTERS" + \
           " WITHOUT FREQUENCY WEIGHTING {:->11s}"
    print(fmt0.format('*', '*'))
    
print("{:-<18s}              GENERATED TOPICS             {:->18s}".
                                                         format('*','*'))
print("{:-<40s}{:->39s}".format('*', '*'))
text_analysis.display_topics(uv, terms, n_terms=15, word_cloud=True)
    
# Save topic clusters and probabilities to pickle
df_prob     = pd.DataFrame(U)
df_prob.columns = ['prob1', 'prob2','prob3','prob4','prob5',
                   'prob6','prob7','prob8', 'prob9']
df          = df.join(df_prob)

# Calculate the topic assignment for each article using the probabilitites
df_topic    = score_topics(U, scores=False)
df['topic'] = df_topic['topic']
#df.sort_values('topic', inplace=True)

# Topic Frequency Table
print("{:28s}{:2s}{:7s}".format("\n   Topic", "N", "Percent"))
print("{:-<19s}{:->19s}".format('*', '*'))
for t in df['topic'].unique():
    lst = df.loc[df['topic'] == t]['source'].value_counts()
    n_source = sum(lst)
    print("   {:<6s}{:.<15d}{:>5d}{:5.1f}%".format(
                "Topic ", t, n_source, 100.0*n_source/n_articles))
print("{:-<19s}{:->19s}".format('*', '*'))

# For each topic, display the frequency of source (news3K vs API)
print("{:28s}{:2s}{:7s}".format("\nTopic/Source", "N", "Percent"))
print("{:-<19s}{:->19s}".format('*', '*'))
for t in df['topic'].unique():
    lst = df.loc[df['topic'] == t]['source'].value_counts()
    n_source = sum(lst)
    print("Topic: ", t)
    for j in range(lst.shape[0]):
        print("     {:.<20s}{:>5d}{:5.1f}%".format(
                lst.index[j], lst[j], 100.0*lst[j]/n_source))
    print("     {:.<20s}{:>5d}{:5.1f}%".format(
                "Topic Total", n_source, 100.0*n_source/n_articles))
print("{:-<19s}{:->19s}".format('*', '*'))

# For each topic, display the frequency of agency
for t in df['topic'].unique():
    lst = df.loc[df['topic'] == t]['agency'].value_counts()
    n_topic = sum(lst)
    print("{:28s}{:2s}{:7s}".format("\nTopic/Agency", "N", "Percent"))
    print("{:-<19s}{:->19s}".format('*', '*'))
    print("Topic: ", t)
    for j in range(lst.shape[0]):
        print("     {:.<20s}{:>5d}{:5.1f}%".format(
                lst.index[j], lst[j], 100.0*lst[j]/n_topic))
    print("     {:.<20s}{:>5d}{:5.1f}%".format(
                "Topic Total", n_topic, 100.0*n_topic/n_articles))
print("{:-<19s}{:->19s}".format('*', '*'))

# Display Frequency Table for "source" (News3K vs API)
pd_series = df['source'].value_counts()
print("{:23s}{:2s}{:7s}".format("\nSource", "N", "Percent"))
print("{:-<17s}{:->17s}".format('*', '*'))
for i in range(pd_series.shape[0]):
    print("{:.<20s}{:>5d}{:5.1f}%".format(
        pd_series.index[i], pd_series[i], 100.0*pd_series[i]/n_articles))
print("{:-<17s}{:->17s}".format('*', '*'))
    
pd_series = df['agency'].value_counts()
print("{:23s}{:2s}{:7s}".format("\nAgency", "N", "Percent"))
print("{:-<17s}{:->17s}".format('*', '*'))
for i in range(pd_series.shape[0]):
    print("{:.<20s}{:>5d}{:5.1f}%".format(
        pd_series.index[i], pd_series[i], 100.0*pd_series[i]/n_articles))
print("{:-<17s}{:->17s}".format('*', '*'))

print("{:23s}{:2s}{:7s}".format("\nAgency/Source", "N", "Percent"))
print("{:-<17s}{:->17s}".format('*', '*'))
for i in range(pd_series.shape[0]):
    print("{:.<20s}{:>5d}{:5.1f}%".format(
        pd_series.index[i], pd_series[i], 100.0*pd_series[i]/n_articles))
    lst = df.loc[df['agency'] == pd_series.index[i]]['source'].value_counts()
    for j in range(lst.shape[0]):
        print("     {:.<15s}{:>5d}{:5.1f}%".format(
                lst.index[j], lst[j], 100.0*lst[j]/n_articles))
print("{:-<17s}{:->17s}".format('*', '*'))

    
filename = "PICKLE/"+Analysis+"_"+today+"_df_TA_pickle.pkl"
df.to_pickle(filename)
print("")
print("{:-<40s}{:->39s}".format('*', '*'))
print("{:-<22s}     TOPIC ANALYSIS COMPLETED     {:->23s}".format('*', '*'))
print("{:-<40s}{:->39s}".format('*', '*'))
