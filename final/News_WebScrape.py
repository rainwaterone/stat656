#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2020

@author: eJones
"""

import pandas as pd
import warnings
from   AdvancedAnalytics.Internet import scrape
from   newsapi.newsapi_client   import NewsApiClient

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

run_news3k = True
if run_news3k:
    words    = ['trump', 'biden', 'democrats', 'republicans']
    filename = "election_7_30_news3k.xlsx"
        
    # dfArticles Columns: agency, url, length, keywords, title, summary, text
    # dfArticles = scrape.newspaper_stories(words, search_type='or', 
    #                                       search_level=1, urls=myUrls, 
    #                                       display=True)
    

    dfArticles = scrape.newspaper_stories(words, search_type='or', 
                                          search_level=0, 
                                          display=True)
    

    # Write to csv to prevent truncation of articles longer than 32767

    dfArticles.sort_values('text', inplace=True)
    dfArticles.drop_duplicates('text', inplace=True)
    dfArticles['source'] = "News3k" # Source Column added to dfArticles
    dfArticles.to_excel(filename, index=False)
    print("Saved", dfArticles.shape[0], "articles to "+filename)

run_newsapi = True
if run_newsapi:
    # Set API Key - get one at https://newsapi.org/pricing
    # It's free to Developers limited to 500 api calls/day
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
    
    words    = ['trump', 'biden', 'democrats', 'republicans']
    filename = 'election_7_30_api.xlsx'
    
    df_urls = scrape.newsapi_get_urls(apikey, words, urls='top_news')
    # Download Discovered Pages
    df_www  = scrape.request_pages(df_urls)
    
    print("Pages Downloaded:", df_www.shape[0])
    df_www.sort_values('text', inplace=True)
    df_www.drop_duplicates(subset='text', inplace=True)
    print("Pages after dropping duplicates:", df_www.shape[0])
    df_www.to_excel(filename,  index=False)
    print("Saved", df_www.shape[0], "articles to "+filename)

update_data = False
# Example code for merging/updating two scrapes
if update_data:
    filename1  = 'file1.xlsx'
    filename2  = 'file2.xlsx'
    df1 = pd.read_excel(filename1)
    df2 = pd.read_excel(filename2)
    
    df = df1.append(df2)
    df.sort_values('text', inplace=True)
    df.drop_duplicates(subset='text', inplace=True)
    df.to_excel("file_combined.xlsx")
