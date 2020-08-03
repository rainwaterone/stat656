#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2020

@author: eJones
"""

import pandas as pd
import warnings
from AdvancedAnalytics.Internet import scrape
from newsapi import NewsApiClient # Needed for using API Feed
import os
import tldextract
from datetime import datetime

filepath = 'C:/Users/rainwater-e/OneDrive - Texas A&M University/Summer-2020/stat656/final/'


def merge_files(filetype, filepath):
    typedict ={'api': 'api.xlsx', 'news3k': 'news3k.xlsx'}
    directory = os.fsencode(filepath) # Encode the directory for iteration
    df_merge = pd.DataFrame()
    
    print('\nReading files...')
    for file in os.listdir(directory):
        filename = os.fsdecode(file) # Get the name of the current file
        if filename.endswith(typedict.get(filetype)):
            filespec = filepath + filename
            print('Reading ',filename,'...')
            df_merge = df_merge.append(pd.read_excel(filespec))
            continue
        else:
            continue
    tot_records = df_merge.shape[0]
    print('Total records: ',tot_records)
    print('Sorting combined dataframe...')
    df_merge.sort_values('text', inplace=True)
    print('Dropping duplicates...')
    df_merge.drop_duplicates(subset='text', inplace=True)
    fin_records = df_merge.shape[0]
    print('Duplicate records dropped: ',tot_records - fin_records)
    print('Final number of records: ',fin_records)
    print('Writing combined file...')
    df_merge.to_excel(filepath + filetype + "_combined.xlsx")

def merge_api_news3k(filepath):
    # df_comb = pd.DataFrame()
    # df_comb.append(pd.read_excel(filepath + 'api_combined.xlsx'))
    # df_comb.append(pd.read_excel(filepath + 'news3k_combined.xlsx'))
    df_api = pd.read_excel(filepath + 'api_combined.xlsx')
    df_news3k = pd.read_excel(filepath + 'news3k_combined.xlsx')
    df_api['agency'] = df_news3k['url'].apply(lambda url: tldextract.extract(url).domain)
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
        'journalstar': 'journalstar'
        }

    
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
    
    run_news3k = True
    if run_news3k:
        words    = ['trump', 'biden', 'democrats', 'republicans']
        filename = "election_" + timestamp + "_news3k.xlsx"
            
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
        filename = 'election_' + timestamp + '_api.xlsx'
        
        df_urls = scrape.newsapi_get_urls(apikey, words, urls='top_news')
        # Download Discovered Pages
        df_www  = scrape.request_pages(df_urls)
        
        print("Pages Downloaded:", df_www.shape[0])
        df_www.sort_values('text', inplace=True)
        df_www.drop_duplicates(subset='text', inplace=True)
        print("Pages after dropping duplicates:", df_www.shape[0])
        df_www.to_excel(filename,  index=False)
        print("Saved", df_www.shape[0], "articles to "+filename)
    

merge_api = True
merge_news3k = True

if merge_api:
    merge_files('api')
    
if merge_news3k:
    merge_files('news3k')


if __name__ == '__main__':
    main()