#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2020

@author: eJones
"""

import pandas as pd
import warnings
from   AdvancedAnalytics.Internet import scrape
from   datetime  import date

try:
    import newspaper # install using conda install newspaper3k
    from   newspaper import Article
except:
    warnings.warn("AdvancedAnalytics.Scrape.newspaper_stories "+\
                  "missing NEWSPAPER3K package")
try:
# newsapi requires tiny\segmenter:  pip install tinysegmenter==0.3
# Install newsapi using:  pip install newsapi-python
    from newsapi import NewsApiClient # Needed for using API Feed
except:
    warnings.warn("AdvancedAnalytics.Scrape.newsapi_get_urls "+\
                  "missing NEWSAPI package")
        
today = str(date.today())
today = today[5:]
# newsapi installed with pip install newsapi-python
# This is the 'API' news feed
# from newsapi import NewsApiClient 
apikey = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

myUrls = {
    'abc-news':            'https://abcnews.go.com',
    'al-jazeera-english':  'http://www.aljazeera.com',
    'associated-press':    'https://apnews.com/',
    'axios':               'https://www.axios.com',
    'bloomberg':           'http://www.bloomberg.com',
    'breitbart-news':      'http://www.breitbart.com',
    'business-insider':    'http://www.businessinsider.com',
    'cbs-news':            'http://www.cbsnews.com',
    'cnbc':                'http://www.cnbc.com',
    'cnn':                 'http://us.cnn.com',
    'engadget':            'https://www.engadget.com',
    'fortune':             'http://fortune.com',
    'fox-news':            'http://www.foxnews.com',
    'google-news':         'https://news.google.com',
    'hill':                'http://thehill.com',
    'huffington-post':     'http://www.huffingtonpost.com',
    'mashable':            'http://mashable.com',
    'medical-news-today':  'http://www.medicalnewstoday.com',
    'msnbc':               'http://www.msnbc.com',
    'national-geographic': 'http://news.nationalgeographic.com',
    'national-review':     'https://www.nationalreview.com/',
    'nbc-news':            'http://www.nbcnews.com',
    'newsweek':            'http://www.newsweek.com',
    'new-york-magazine':   'http://nymag.com',
    'politico':            'https://www.politico.com',
    'reddit-r-all':        'https://www.reddit.com/r/all',
    'reuters':             'http://www.reuters.com',
    'usa-today':           'http://www.usatoday.com/news',
    'verge':               'http://www.theverge.com',
    'washington-post':     'https://www.washingtonpost.com'
}

# Increase column width to let pandy read large text columns
pd.set_option('max_colwidth', 100000)
w_elect  = ['trump', 'biden', 'democrats', 'republicans']
w_virus  = ['covid', 'coronavirus']
f_elect_news3k  = "Final2020/Nov2020_Elections_m_dd_news3k.xlsx"
f_virus_news3k  = "Final2020/Covid_m_dd_news3k.xlsx"
f_elect_api     = "Final2020/Nov2020_Elections_m_dd_api.xlsx"
f_virus_api     = "Final2020/Covid_m_dd_api.xlsx"
f_elect_news3k  = f_elect_news3k.replace('m_dd', today)
f_virus_news3k  = f_virus_news3k.replace('m_dd', today)
f_elect_api     = f_elect_api.replace('m_dd', today)
f_virus_api     = f_virus_api.replace('m_dd', today)

run_news3k = False
if run_news3k:
    words    = w_virus
    filename = f_virus_news3k
    # dfArticles Columns: agency, url, length, keywords, title, summary, text
    dfArticles = scrape.newspaper_stories(words, search_type='or', 
                                          search_level=1, urls=myUrls,
                                          display=True)
    
    # Write to csv to prevent truncation of articles longer than 32767
    dfArticles.sort_values('url', inplace=True)
    dfArticles.drop_duplicates(subset=['url'], inplace=True)
    dfArticles['source'] = "News3k" # Source Column added to dfArticles
    dfArticles.to_excel(filename, index=False)
    print("Saved", dfArticles.shape[0], "articles to "+filename)

run_newsapi = False
if run_newsapi:
    words    = w_virus
    filename = f_virus_api
    # Set API Key - get one at https://newsapi.org/pricing
    # It's free to Developers limited to 300 api calls/day
    news_list = {}
    newsapi = NewsApiClient(apikey)
    
    # Use the following code to get the current list
    # of agencies used by newsapi
    sources   = newsapi.get_sources()
    n_sources = len(sources['sources'])
    
    for i in range(n_sources):
        cay  = sources['sources'][i]['id']
        val  = sources['sources'][i]['url']
        lang = sources['sources'][i]['language']
        ctry = sources['sources'][i]['country']
        if lang == 'en' and ctry == 'us':
            news_list[cay] = val
    
    df_urls = scrape.newsapi_get_urls(apikey, words, urls=myUrls)
    # Download Discovered Pages
    df_www  = scrape.request_pages(df_urls)
    
    print("Pages Downloaded:", df_www.shape[0])
    df_www.sort_values('url', inplace=True)
    df_www.drop_duplicates(subset=['url'], inplace=True)
    print("Pages after dropping duplicates:", df_www.shape[0])
    df_www.to_excel(filename,  index=False)
    print("Saved", df_www.shape[0], "articles to "+filename)

update_data = True
# Example code for merging/updating two scrapes
if update_data:
    filename1    = 'Final2020/Covid_08-03_news3k.xlsx'
    filename2    = 'Final2020/Covid_8_01_news3k_all.xlsx'
    filename_out = 'Final2020/Covid_8_03_news3k_all.xlsx'
    df1 = pd.read_excel(filename1)
    df2 = pd.read_excel(filename2)
    df  = df1.append(df2)
    
    dropped = df.shape[0]
    df.sort_values('url', inplace=True)
    df.drop_duplicates(subset='url', inplace=True)
    df.to_excel(filename_out, index=False)
    dropped = dropped - df.shape[0]
    print("\n {:<40s}{:>11s}".format("Filename", "Articles"))
    print("*{:-<51s}*".format("-"))
    print(" {:.<45s}{:>5d}".format(filename1[10:],df1.shape[0]))
    print(" {:.<45s}{:>5d}".format(filename2[10:],df2.shape[0]))
    print(" {:.<45s}{:>5d}".format(filename_out[10:],df.shape[0]))
    print("*{:-<51s}*".format("-"))
    print("Saved", df.shape[0], "articles and Dropped", dropped)
    
