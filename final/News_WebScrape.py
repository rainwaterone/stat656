#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAT 656 Final
@author:Lee Rainwater
@heavy_lifting_by: Dr. Edward Jones
@date: 2020-08-05

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
from AdvancedAnalytics.Internet import scrape
from newsapi import NewsApiClient  # Needed for using API Feed
import os
import tldextract
from AdvancedAnalytics.Text import text_analysis
from AdvancedAnalytics.Text import sentiment_analysis
from sklearn.feature_extraction.text import CountVectorizer
from AdvancedAnalytics.Text import text_plot
import numpy as np
from collections import Counter
from datetime import datetime

combfile = "grandUnified.xlsx"
filepath = 'C:/Users/rainwater-e/OneDrive - Texas A&M University/Summer-2020/\
stat656/final/'
search_terms = ['trump', 'biden', 'democrats', 'republicans']

text_col = 'text'
groupcol = 'agency'


def heading(headerstring):
    """
    Centers headerstring on the page. For formatting to stdout.

    Parameters
    ----------
    headerstring : string
    String that you wish to center.
    Returns
    -------
    Returns: None.
    """
    tw = 70  # text width
    lead = int(tw/2)-(int(len(headerstring)/2))-1
    tail = tw-lead-len(headerstring)-2
    print('\n' + ('*'*tw))
    print(('*'*lead) + ' ' + headerstring + ' ' + ('*'*tail))
    print(('*'*tw))
    return


def sentiment(inputfile, filepath, text_col, groupcol, search_terms):
    """
    Generate sentiment analysis.

    Returns
    -------
    None.

    From HW 10 assignment, replaced 'hotel' with groupcol.

    """
    terms_of_interest = ['trump', 'biden', 'fauci']
    heading("READING DATA SOURCE...")

    # Set Pandas Columns Width for Excel Columns
    pd.set_option('max_colwidth', 32000)
    df = pd.read_excel(filepath + inputfile, usecols=[groupcol, text_col])
    text = np.array(df[text_col])
    # Check if any text was truncated
    pd_width = pd.get_option('max_colwidth')
    # Maps text_col onto len() and finds max()
    maxsize = df[text_col].map(len).max()
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
    cv = CountVectorizer(max_df=1.0, min_df=1, max_features=None,
                         ngram_range=(1, 2), analyzer=sa.analyzer,
                         vocabulary=sa.sentiment_word_dic)
    stf = cv.fit_transform(text)  # Return document-term matrix
    tf = cv.transform(df[text_col])  # Create term-frequency matrix
    sterms = cv.get_feature_names()  # Map feature indices to feature names
    heading("CALCULATE AND STORE SENTIMENT SCORES...")
    # Calculate and Store Sentiment Scores into DataFrame "s_score"
    s_score = sa.scores(stf, sterms)
    n_reviews = s_score.shape[0]
    n_sterms = s_score['n_words'].sum()
    max_length = df[text_col].apply(len).max()
    if n_sterms == 0 or n_reviews == 0:
        print("No sentiment terms found.")

    p = s_score['n_words'].sum() / n_reviews
    print('{:-<24s}{:>6d}'.format("\nMaximum Text Length", max_length))
    print('{:-<23s}{:>6d}'.format("Total Reviews", n_reviews))
    print('{:-<23s}{:>6d}'.format("Total Sentiment Terms", n_sterms))
    print('{:-<23s}{:>6.2f}'.format("Avg. Sentiment Terms", p))

    # s_score['sentiment'] = s_score['sentiment'].map("{:,.2f}".format)
    df = df.join(s_score)
    print("\n", df[[groupcol, 'sentiment', 'n_words']], "\n")

    print(df.groupby([groupcol]).mean())

    # count the number of terms of interest in each document
    for term in terms_of_interest:
        df[text_col] = df[text_col].str.lower()
        df[term] = df[text_col].str.count(term)

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(df.trump, df.sentiment, s=10, c='r', label='Trump')
    ax1.scatter(df.biden, df.sentiment, s=10, c='b', label='Biden')
    ax1.scatter(df.fauci, df.sentiment, s=10, c='g', label='Fauci')
    plt.legend(loc='upper right')
    plt.xlabel('Mentions')
    plt.ylabel('Sentiment')
    plt.title('Sentiment vs. Article Mentions')
    plt.show()

    # display_term_frequency(stf, sterms, word_cloud=False, zipf_plot=False)

    # heading("GENERATING TOTAL WORD CLOUD FOR CORPUS...")
    # tcv = CountVectorizer(max_df=1.0, min_df=1, max_features=None,
    #                       ngram_range=(1, 2), analyzer=ta.analyzer)
    # tf = tcv.fit_transform(df[text_col])
    # terms = tcv.get_feature_names()
    # td = text_plot.term_dic(tf, terms)
    # text_plot.word_cloud_dic(td, max_words=200)

    # heading("GENERATING SENTIMENT WORD CLOUD FOR CORPUS...")
    # corpus_sentiment = {}
    # n_sw = 0
    # for i in range(n_reviews):
    #     # Iterate over the terms with nonzero scores."stf" is a sparse matrix
    #     term_list = stf[i].nonzero()[1]
    #     if len(term_list) > 0:
    #         for t in np.nditer(term_list):
    #             score = sa.sentiment_dic.get(sterms[t])
    #             if score is not None:
    #                 n_sw += stf[i, t]
    #                 current_count = corpus_sentiment.get(sterms[t])
    #                 if current_count is not None:
    #                     corpus_sentiment[sterms[t]] = stf[i, t]
    #                 else:
    #                     corpus_sentiment[sterms[t]] += stf[i, t]

    # # Word cloud for the Sentiment Words found in the Corpus
    # text_plot.word_cloud_dic(corpus_sentiment, max_words=200)
    # n_usw = len(corpus_sentiment)
    # print("\nSENTIMENT TERMS")
    # print("------------------")
    # print("{:.<10s}{:>8d}".format("Unique", n_usw))
    # print("{:.<10s}{:>8d}".format("Total", n_sw))
    # print("------------------")

    # heading("GENERATING TOTAL WORD CLOUD FOR BELLAGIO...")
    # tcv = CountVectorizer(max_df=1.0, min_df=1, max_features=None,
    #                       ngram_range=(1, 2), analyzer=ta.analyzer)
    # tf = tcv.fit_transform(df[df[groupcol] == 'Bellagio'][text_col])
    # terms = tcv.get_feature_names()
    # td = text_plot.term_dic(tf, terms)
    # text_plot.word_cloud_dic(td, max_words=200)

    # heading("GENERATING SENTIMENT WORD CLOUD FOR BELLAGIO...")

    # bcv = CountVectorizer(max_df=1.0, min_df=1, max_features=None,
    #                       ngram_range=(1, 2), analyzer=sa.analyzer,
    #                       vocabulary=sa.sentiment_word_dic)
    # # Return document-term matrix
    # bstf = bcv.fit_transform(df[df[groupcol] == 'Bellagio'][text_col])
    # bsterms = bcv.get_feature_names()  # Map feature indices to feature names
    # heading("CALCULATE AND STORE SENTIMENT SCORES FOR BELLAGIO...")
    # # Calculate and Store Sentiment Scores into DataFrame "s_score"
    # bs_score = sa.scores(bstf, bsterms)
    # bn_reviews = bs_score.shape[0]
    # bn_sterms = bs_score['n_words'].sum()
    # max_length = df['Review'].apply(len).max()
    # if bn_sterms == 0 or bn_reviews == 0:
    #     print("No sentiment terms found.")

    # corpus_sentiment = {}
    # n_sw = 0
    # for i in range(bn_reviews):
    #     # Iterate over the terms with nonzero scores."stf" is a sparse matrix
    #     term_list = bstf[i].nonzero()[1]
    #     if len(term_list) > 0:
    #         for t in np.nditer(term_list):
    #             score = sa.sentiment_dic.get(bsterms[t])
    #             if score is not None:
    #                 n_sw += bstf[i, t]
    #                 current_count = corpus_sentiment.get(bsterms[t])
    #                 if current_count is not None:
    #                     corpus_sentiment[bsterms[t]] = bstf[i, t]
    #                 else:
    #                     corpus_sentiment[bsterms[t]] += bstf[i, t]

    # # Word cloud for the Sentiment Words found in the Corpus
    # text_plot.word_cloud_dic(corpus_sentiment, max_words=200)
    # n_usw = len(corpus_sentiment)
    # print("\nBELLAGIO SENTIMENT TERMS")
    # print("------------------")
    # print("{:.<10s}{:>8d}".format("Unique", n_usw))
    # print("{:.<10s}{:>8d}".format("Total", n_sw))
    # print("------------------")
    return


def display_term_frequency(tf, terms, n_tail=20, tfidf=None, word_cloud=True,
                           zipf_plot=True):
    td = text_plot.term_dic(tf, terms, scores=None)
    # Calculate term cdf
    k = Counter(td)
    sorted_terms = k.most_common()
    n_terms = len(sorted_terms)
    doc_terms = {}
    tfidf_score = {}
    for i in range(n_terms):
        # Store term document-frequency in doc_terms
        doc_terms[terms[i]] = tf[:, i].count_nonzero()
    if tfidf is not None:
        for i in range(n_terms):
            tfidf_score[terms[i]] = tfidf[:, i].sum()
    # Display the top 20 terms
    k = Counter(td)
    tail_terms = k.most_common(n_tail)
    print("")
    print(n_tail, "MOST COMMON TERMS")
    if tfidf is None:
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
    if tfidf is None:
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
                                 bg_color="maroon", size=(400, 200),
                                 random=12345)
        print("")

    if zipf_plot is True:
        # Standard ZIFF plot using log(term frequency) on the vertical axis
        freq = np.zeros(n_terms)
        i = 0
        for t in sorted_terms:
            freq[i] = t[1]
            i += 1
        plt.figure(figsize=(9, 4))
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
            i += 1
        plt.figure(figsize=(9, 4))
        plt.title('Term Frequency for Most Common Terms')
        plt.grid(True)
        plt.yscale('log')
        plt.ylabel("Log(Term Frequency)")
        plt.xticks(rotation=45, ha='right')
        ax = plt.gca()
        ax.set_facecolor('steelblue')
        plt.bar(x, y, color='gold')
        plt.show()


def merge_files(filetype, filepath):
    """
    Given a specified news file type ('api' or 'news3k'), merges the daily
    files of that type into a single file of name:
        api    - 'api_combined.xlsx'
        news3k - 'news3k_combined.xlsx'

    Parameters
    ----------
    filetype : STRING
        The value 'api' or 'news3k'
    filepath : STRING
        The filesystem location of the input files. The output file will be
        written at the same location.

    Returns
    -------
    None.

    """
    typedict = {'api': 'api.xlsx', 'news3k': 'news3k.xlsx'}
    directory = os.fsencode(filepath)  # Encode the directory for iteration
    df_merge = pd.DataFrame()

    print('\nReading ', filetype, ' files...')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)  # Get the name of the current file
        if filename.endswith(typedict.get(filetype)):
            filespec = filepath + filename
            print('Reading ', filename, '...')
            df_merge = df_merge.append(pd.read_excel(filespec))
            continue
        else:
            continue
    tot_records = df_merge.shape[0]
    print('Total records: ', tot_records)
    print('Sorting combined dataframe...')
    df_merge.sort_values('text', inplace=True)
    print('Dropping duplicates...')
    df_merge.drop_duplicates(subset='text', inplace=True)
    fin_records = df_merge.shape[0]
    print('Duplicate records dropped: ', tot_records - fin_records)
    print('Final number of records: ', fin_records)
    print('Writing combined file...')
    df_merge['source'] = filetype
    df_merge.to_excel(filepath + filetype + "_combined.xlsx", index=False)
    return


def merge_api_news3k(outfile, filepath):
    """
    Read the files api_combined.xlsx and news3k_combined.xlsx that are in
    filepath into dataframes. The API 'agency' field is extracted from the URL
    and mapped to the equivalent news3k agency via dictionary.
    Parameters
    ----------
    filepath : STRING
        The filesystem location of the input files. The output file will be
        written at the same location.

    Returns
    -------
    None.

    """
    df_api = pd.read_excel(filepath + 'api_combined.xlsx')
    df_news3k = pd.read_excel(filepath + 'news3k_combined.xlsx')
    # Parse the URL to extract the domain name as the agency
    print('\nExtracting domain name from API files to use as agency...')
    df_api['agency'] = df_api['url'].apply(lambda
                                           url: tldextract.extract(url).domain)
    # Create a dictionary to map api agencies to equivalent news3k agencies
    agency_dict = {
        'foxnews': 'fox',
        'cnn': 'cnn',
        'npr': 'npr',
        'reuters': 'reuters',
        'nbcnews': 'nbc-news',
        'cbsnews': 'cbs-news',
        'msnbc': 'msnbc',
        'usatoday': 'usa-today',
        'usnews': 'us-news',
        'huffpost': 'huffington',
        'go': 'go',
        'abcnews': 'abc-news',
        'huffingtonpost': 'huffington',
        'whitehouse': 'whitehouse',
        'msn': 'msn',
        'journalstar': 'journalstar',
        'apnews': 'apnews',
        'aljazeera': 'aljazeera',
        'bloomberg': 'bloomberg',
        'wsj': 'wsj'
        }
    print('Harmonizing API agency names with News3k agency names...')
    df_api['agency'] = df_api['agency'].map(agency_dict)
    print('Combining into ', outfile, '...', end='')
    df_comb = df_news3k.append(df_api, ignore_index=True)
    df_comb.to_excel(filepath + "grandUnified.xlsx", index=False)
    print('...complete.')

    return


def main():
    starttime = datetime.now()
    timestamp = starttime.strftime('%Y-%m-%d-%H_%M_%S')
    # try:
        # import newspaper # install using conda install newspaper3k
        # from   newspaper import Article
    # except:
    #     warnings.warn("AdvancedAnalytics.Scrape.newspaper_stories "+\
                      # "missing NEWSPAPER3K package")
    # try:
    # newsapi requires tiny\segmenter:  pip install tinysegmenter==0.3
    # Install newsapi using:  pip install newsapi-python
    # except:
    #     warnings.warn("AdvancedAnalytics.Scrape.newsapi_get_urls "+\
    #                   "missing NEWSAPI package")
    # newsapi installed with pip install newsapi-python
    # This is the 'API' news feed
    # from newsapi import NewsApiClient

    apikey = 'd5009ffb61ca46568c5ad6c52ddf49e9'

    myUrls = {
            'new-scientist': 'https://www.newscientist.com/section/news',
            'medical-news-today': 'http://www.medicalnewstoday.com',
            'hacker-news':        'https://news.ycombinator.com',
            'associated-press':   'https://apnews.com/'
        }
    # Increase column width to let pandy read large text columns
    pd.set_option('max_colwidth', 100000)

    run_news3k = False
    if run_news3k:
        filename = "election_" + timestamp + "_news3k.xlsx"

        # dfArticles Columns: agency, url, length, keywords, title, summary, text
        # dfArticles = scrape.newspaper_stories(words, search_type='or', 
        #                                       search_level=1, urls=myUrls, 
        #                                       display=True)

        dfArticles = scrape.newspaper_stories(search_terms, search_type='or',
                                              search_level=0,
                                              display=True)

        # Write to csv to prevent truncation of articles longer than 32767
        dfArticles.sort_values('text', inplace=True)
        dfArticles.drop_duplicates('text', inplace=True)
        # dfArticles['source'] = "News3k"  # Source Column added to dfArticles
        dfArticles.to_excel(filename, index=False)
        print("Saved", dfArticles.shape[0], "articles to "+filename)

    run_newsapi = False
    if run_newsapi:
        # Set API Key - get one at https://newsapi.org/pricing
        # It's free to Developers limited to 500 api calls/day
        news_list = {}
        newsapi = NewsApiClient(apikey)

        # Use the following code to get the current list
        # of agencies used by newsapi
        sources = newsapi.get_sources()
        n_sources = len(sources['sources'])

        for i in range(n_sources):
            cay = sources['sources'][i]['id']
            val = sources['sources'][i]['url']
            lang = sources['sources'][i]['language']
            ctry = sources['sources'][i]['country']
            if lang == 'en' and ctry == 'us':
                news_list[cay] = val

        filename = 'election_' + timestamp + '_api.xlsx'

        df_urls = scrape.newsapi_get_urls(apikey, search_terms, urls='top_news')
        # Download Discovered Pages
        df_www = scrape.request_pages(df_urls)

        print("Pages Downloaded:", df_www.shape[0])
        df_www.sort_values('text', inplace=True)
        df_www.drop_duplicates(subset='text', inplace=True)
        print("Pages after dropping duplicates:", df_www.shape[0])
        df_www.to_excel(filename,  index=False)
        print("Saved", df_www.shape[0], "articles to "+filename)

    merge_api = False      # Set True to merge daily API files
    merge_news3k = False   # Set True to merge daily News3k files
    merge_sources = False  # Set True to merge combined API and News3k files
    run_sentiment = True  # Set True to run sentiment analysis

    if merge_api:
        merge_files('api', filepath)

    if merge_news3k:
        merge_files('news3k', filepath)

    if merge_sources:
        merge_api_news3k(combfile, filepath)

    if run_sentiment:
        sentiment(combfile, filepath, text_col, groupcol, search_terms)
    return


if __name__ == '__main__':
    main()
