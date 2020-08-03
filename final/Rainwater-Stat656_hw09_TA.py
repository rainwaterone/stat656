import sys, math, pickle
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
# Classes provided for the course
from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT
from AdvancedAnalytics.Regression          import linreg, stepwise
from AdvancedAnalytics.Text import text_analysis, text_plot
# Packages from Sci-Learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition   import LatentDirichletAllocation
from sklearn.decomposition   import TruncatedSVD
from sklearn.decomposition   import NMF
from sklearn.model_selection import train_test_split
# Other packages
from collections import Counter

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


def display_term_frequency(tf, terms, n_tail=20, tfidf=None, word_cloud=True,
                           zipf_plot=True):
    td = text_plot.term_dic(tf, terms, scores=None)
    # Calculate term cdf
    k            = Counter(td)
    sorted_terms = k.most_common()
    n_terms      = len(sorted_terms)
    doc_terms    = {}
    tfidf_score  = {}
    for i in range(n_terms):
        # Store term document-frequency in doc_terms
        doc_terms[terms[i]] = tf[:,i].count_nonzero()
    if tfidf != None:
        for i in range(n_terms):
            tfidf_score[terms[i]] = tfidf[:,i].sum()
    # Display the top 20 terms
    k          = Counter(td)
    tail_terms = k.most_common(n_tail)
    print("")
    print(n_tail, "MOST COMMON TERMS")
    if tfidf  == None:
        print("---------------------------------------")
        print("{:.<15s}{:>12s}{:>12s}".format('Term', 'Term Freq.', 
                                              'Doc Freq.'))
        for t in tail_terms:
            # Uset his formation when unweighted frequencies are used
            print("{:.<15s}{:>9d}{:>12d}".format(t[0], t[1], doc_terms[t[0]]))
        print("---------------------------------------\n")
    else:
        print("------------------------------------------------")
        print("{:.<15s}{:>12s}{:>12s}{:>9s}".format('Term', 'Term Freq.', 
                                              'Doc Freq.', 'TFIDF'))
        for t in tail_terms:
            # Uset his formation when unweighted frequencies are used
            print("{:.<15s}{:>9d}{:>12d}{:>12.1f}".format(t[0], t[1], 
                                                         doc_terms[t[0]],
                                                         tfidf_score[t[0]]))
        print("------------------------------------------------\n")
    
    bot_terms = k.most_common()[-n_tail:]
    print(n_tail, "LEAST COMMON TERMS")
    if tfidf  == None:
        print("---------------------------------------")
        print("{:.<15s}{:>12s}{:>12s}".format('Term', 'Term Freq.', 
                                              'Doc Freq.'))
        for t in bot_terms:
            # Uset his formation when unweighted frequencies are used
            print("{:.<15s}{:>9d}{:>12d}".format(t[0], t[1], doc_terms[t[0]]))
        print("---------------------------------------\n")
    else:
        print("------------------------------------------------")
        print("{:.<15s}{:>12s}{:>12s}{:>9s}".format('Term', 'Term Freq.', 
                                              'Doc Freq.', 'TFIDF'))
        for t in bot_terms:
            # Uset his formation when unweighted frequencies are used
            print("{:.<15s}{:>9d}{:>12d}{:>12.1f}".format(t[0], t[1], 
                                                         doc_terms[t[0]],
                                                         tfidf_score[t[0]]))
        print("------------------------------------------------\n")
        
    if word_cloud:
        # Work cloud for top terms - terms with highest term frequency
        text_plot.word_cloud_dic(td, mask=None, max_words=n_tail, 
                                 bg_color="maroon", size=(400,200), 
                                 random=12345)
        print("")
    
    if zipf_plot==True:
        # Standard ZIFF plot using log(term frequency) on the vertical axis
        freq  = np.zeros(n_terms)
        i = 0
        for t in sorted_terms:
            freq [i] = t[1]
            i += 1
        plt.figure(figsize=(9,4))
        plt.title('Log ZIPF Plot')
        plt.xlabel("Rank")
        plt.ylabel("Log(Term Frequency)")
        plt.yscale('log')
        plt.title('Log Zipf Plot')
        plt.grid(True)
        ax = plt.gca()
        ax.set_facecolor('steelblue')
        plt.plot(freq, '-', color='gold', linewidth=3)
        plt.show()
        
        x = np.zeros(n_tail).astype(str)
        y = np.zeros(n_tail)
        i = 0
        for t in tail_terms:
            x[i] = t[0]
            y[i] = t[1]
            i+=1
        plt.figure(figsize=(9,4))
        plt.title('Term Frequency for Most Common Terms')
        plt.grid(True)
        plt.yscale('log')
        plt.ylabel("Log(Term Frequency)")
        plt.xticks(rotation=45, ha='right')
        ax = plt.gca()
        ax.set_facecolor('steelblue')
        plt.bar(x, y, color='gold')
        plt.show()
    


#*****************************************************************************
# Increase column width to let pandy read large text columns
pd.set_option('max_colwidth', 32000)
print("{:-<40s}{:->39s}".format('*', '*'))
print("{:-<21s}     TEXT ANALYSIS OF NTHSA HONDA    {:->21s}".format('*', '*'))
print("{:-<22s}            CRASH DATA             {:->22s}".format('*', '*'))
print("{:-<40s}{:->39s}".format('*', '*'))
# Read N=11,717 California Chardonnay Reviews
df = pd.read_excel("HondaComplaints.xlsx")
n_reviews    = df.shape[0]# Check to see if documents (descriptions) were truncated
maxsize      = 0
n_truncated  = 0
pd_width     = pd.get_option('max_colwidth')

# Find maximum record length and count number that are truncated
for i in range(n_reviews):
    if len(df['description'].iloc[i]) >= pd_width:
        n_truncated += 1
    if len(df['description'].iloc[i]) > maxsize:
        maxsize = len(df['description'].iloc[i])
print("\nTEXT COLUMN WIDTH:")
print("{:<17s}{:>6d}".format("   Max. Accepted", pd_width))
print("{:<17s}{:>6d}".format("   Max. Observed", maxsize))
if n_truncated>0:
    print("{:<17s}{:>6d}".format("   Truncated", pd_width))
else:
    print("{:<17s}{:>6d}".format("   Truncated", 0))

print("\n{:-<22s}   STARTING TEXT PARSE & FILTER    {:->22s}\n".
      format('*', '*'))
# Setup program constants and reviews
ngram    = (1,2)
text     = 'description' # Dataframe text column
reviews  = np.array(df[text])

# Constants
n_topics  =  8    # Requested number of topic clusters
max_terms = 15    # Max terms to describe in topic clusters
tfidf     = True  # Use TF-IDF Weighting if True

# Create Word Frequency by Review Matrix using Custom Analyzer
# Synonym Dictionary
# The below dictionary seemed to work, except that it failed to synonymize
# 'bag' to airbag.

syn = {"automobile": 'car', 
        "vehicle": 'car', 
        "gasoline": 'gas', 
        "crv" : 'cr-v',
        "cr v": 'cr-v',
        "cr": 'cr-v',
        "fix": 'repair',
        "dealership": 'dealer',
        "honda dealer": 'dealer',
        "manufacturer": 'honda', 
        "replacement":'replace',
        "suddenly": 'sudden',
        "recall campaign":'recall',
        "campaign": 'recall',
        "issue": 'problem',
        "complaint":'problem',
        "highway": 'road', 
        "seat belt":'seatbelt',
        "air bag":'airbag',
        "air  bag":'airbag',
        "airbags": 'airbag',
        "srs":'airbag',
        "sr": 'airbag',
        "bag":'airbag'}


# The below dictionary did not work to synonymize different terms for 'airbag'
# syn = {"automobile": 'car', 
#         "vehicle": 'car', 
#         "gasoline": 'gas', 
#         "crv" : 'cr-v',
#         "dealership": 'dealer',
#         "manufacturer": 'honda', 
#         "issue": 'problem',
#         "highway": 'road', 
#         "seat belt":'seatbelt',
#         "airbag":'air bag',
#         "airbag":'air  bag',
#         "airbag": 'airbags',
#         "airbag":'srs',
#         "airbag":'sr',
#         "airbag":'bag'}

# The below dictionary did not work in assigning a list to the 'air bag' key
# syn = {"automobile": 'car', 
#        "vehicle": 'car', 
#        "gasoline": 'gas', 
#        "crv" : 'cr-v',
#        "dealership": 'dealer',
#        "manufacturer": 'honda', 
#        "issue": 'problem',
#        "highway": 'road', 
#        "seat belt":'seatbelt',
#        "air bag":['airbag', 'srs', 'sr']}


# Additional Stop Words
sw = ['tl', 'tr', 'jb', 'xxx', 'know', 'thing', 'regard', 'provide',
      'car','honda', 'notified', 'acura', 'call', 'tell', 'approximately',
      'read', 'sinc', 'id', 'offer', 'nhtsa', 'unit', 'go', 'many', 'need',
      'make', 'number', 'information', 'cause', 'receive', 'prior', 
      'notice', 'approximate', 'told', 'entire', 'happen',
      'dealer', 'problem', 'contact', 'internet', 'appear', 'year',
      'online', 'find', 'lj', 'responsible', 'consumer', 'part', 'van']
ta = text_analysis(synonyms=syn, stop_words=sw, pos=True, stem=True)

max_df   = 0.98 # max proportion of docs/term allowed
if max_df > 1:
    max_docs = max_df
else:
    max_docs = math.floor(max_df*n_reviews)
print("max_df=",max_df,"suppresses terms appearing in more than", 
      max_docs, "reviews")
min_df   =   5 # min number of docs required to include terms
if min_df > 1:
    min_docs = min_df
else:
    min_docs = math.ceil(min_df*n_reviews)
print("min_df=",min_df,"suppresses terms appearing in less than", 
      min_docs, "reviews")
cv = CountVectorizer(max_df=max_df, min_df=min_df, max_features=None,
                     binary=False,  ngram_range=ngram, analyzer=ta.analyzer)
tf      = cv.fit_transform(reviews)
terms   = cv.get_feature_names()
n_terms = len(terms)
print('{:.<22s}{:>6d}'.format("Number of Complaints", n_reviews))
print('{:.<22s}{:>6d}'.format("Number of Terms",   n_terms))
print("\n{:-<22s}  BUILDING TERM-FREQUENCY MATRIX  {:->23s}".format('*', '*'))

#************************ TERM WEIGHTING *************************************
# Apply TFIDF weighting if tfidf is set to True
if tfidf != True:
    #****** DISPLAY MOST AND LEAST COMMON TERMS WITHOUT TDIDF WEIGHTING ******
    display_term_frequency(tf, terms, n_tail=20)# display top and bottom terms
else:
    # Use TFIDF Weighting
    print("\nModifying Term/Frequency Matrix using TF-IDF")
    # Default for norm is 'l2', use norm=None to supress
    tfidf_vect = TfidfTransformer(norm=None, use_idf=True)
    # tf matrix is (n_reviews)x(m_features)
    tf_idf = tfidf_vect.fit_transform(tf)
    #********* DISPLAY MOST AND LEAST COMMON TERMS WITH TDIDF WEIGHTING ******
    display_term_frequency(tf, terms, n_tail=20, tfidf=tf_idf) 
        
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
    raise RuntimeError("dcomp not equal to lda, svd, nmf or kld")
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
                   'prob6','prob7','prob8']
df          = df.join(df_prob)
df.to_pickle('Honda_df_pickle.pkl')
print("---------------------------------------------------\n")
print("{:-<22s}      TEXT ANALYSIS COMPLETED     {:->23s}".format('*', '*'))
print("{:-<40s}{:->39s}".format('*', '*'))
