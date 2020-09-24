#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 23:06:21 2020

@author: EJones
"""
import sys
import pandas as pd
import numpy  as np
from   datetime  import date

today = str(date.today())
today = today[5:]

min_characters = 300 # Minimum characters required for each article

my_agencies = ['huffington', 'reuters', 'cbs-news', 'usa-today', 'cnn',
            'npr', 'abc-news', 'us-news', 'msn', 'pbs', 'nbc-news',
            'cnbc', 'msnbc', 'fox', 'apnews', 'breitbart', 'al-jazeera',
            'axios', 'washington-times', 'politics', 'business', 
            'sports', 'technology']

tx_univ = ['A&M', 'A&M-Commerce', 'A&M-Corpus', 'A&M-Kingsville',
            'A&M-Galveston', 'A&M-PrairieView', 'A&M-International',
            'A&M-WestTexas', 'Baylor', 'Rice', 'SFAustin', 'SMU',
            'SulRoss', 'TexasState', 'Texas_Tech', 'UDallas', 'UHouston',
            'UTexas', 'UT_Dallas', 'UT_ElPaso', 'UT_Houston', 'UT_NorthTexas',
            'UT_SanAntonio']

API_Agency_Extract = False
if API_Agency_Extract:
    filename_in  = 'Final2020/Nov2020_Elections_8_02_api_all.xlsx'
    filename_out = 'Final2020/Nov2020_Elections_8_02_api_all_agency.xlsx'
    df           = pd.read_excel(filename_in)
    df           = df[['url', 'text']]
    df['source'] = "api"
    df['length'] = 0
    df['trump']  = 0
    df['biden']  = 0
    df['fauci']  = 0
    df['agency'] = "         ";
    for i in range(df.shape[0]):
        if type(df['text'].iloc[i]) != str:
            df['text'].iloc[i] = ""
        df['length'].iloc[i] = len(df['text'].iloc[i])
        ag = df['url'].iloc[i].strip()
        ag = ag.replace("http://",  "")
        ag = ag.replace("https://", "")
        ag = ag.replace("af.",      "")
        ag = ag.replace("ca.",      "")
        ag = ag.replace("in.",      "")
        ag = ag.replace("uk.",      "")
        ag = ag.replace("www.",     "")
        ag = ag.replace("radio.",   "")
        ag = ag.replace("video",    "")
        ag = ag.replace("go.",      "")
        ag = ag.replace(".com",     "")
        ag = ag.replace("markets.", "")
        ag = ag.replace("global.",  "")
        ag = ag.replace("-post",    "")
        j  = ag.find('/')
        ag = ag[0:j]
        ag = ag.replace("cbsnews",        "cbs-news")
        ag = ag.replace("abcnews",        "abc-news")
        ag = ag.replace("usatoday",       "usa-today")
        ag = ag.replace("foxnews",        "fox")
        ag = ag.replace(".fox",           "fox")
        ag = ag.replace("nbcnews",        "nbc-news")
        ag = ag.replace("washingtontimes","washington-times")
        ag = ag.replace("aljazeera",      "al-jazeera")
        ag = ag.replace("businessinsider","business")
        ag = ag.replace("businessinsider","business")
        ag = ag.replace("fortune",        "business")
        ag = ag.replace("nfl",            "sports")
        ag = ag.replace("nhl",            "sports")
        ag = ag.replace("bleacherreport", "sports")
        ag = ag.replace("foxsports",      "sports")
        ag = ag.replace("nationalreview", "politics")
        ag = ag.replace("thehill",        "politics")
        ag = ag.replace("politico",       "politics")
        ag = ag.replace("wired",          "technology")
        ag = ag.replace("techradar",      "technology")
        ag = ag.replace("buzzfeed",       "technology")
        ag = ag.replace("engadget",       "technology")
        ag = ag.replace("arstechnicom",   "technology")
        ag = ag.replace("techcrunch",     "technology")
        ag = ag.replace("verge",          "technology")
        if df['agency'].iloc[i] not in my_agencies:
            #print(df['agency'].iloc[i])
            df['agency'].iloc[i] = "other"
        
        df['agency'].iloc[i] = ag.strip()
        # Count number of times trump, biden and fauci appear in articles
        txt = df['text'].iloc[i].lower()
        df['trump'].iloc[i] = txt.count('trump')
        df['biden'].iloc[i] = txt.count('biden')
        df['fauci'].iloc[i] = txt.count('fauci')
    print("")
    df = df[df.length>min_characters]
    for ag in np.sort(df['agency'].unique()):
        n = df.shape[0]
        t = sum(df['agency']==ag)
        r = 100*t/n
        print("{:.<18s} {:>4d} {:>3.0f}%".format(ag, t, r))
    print("{:.<18s} {:>4d} {:>3.0f}%".format("TOTAL", n, 100.0))
    print("\nSaving-->",filename_out)
    df3 = df.copy(deep=True)
    df3.to_excel(filename_out, index=False)
    print("Columns Saved: ", df.columns)
    print("Top 20 Agencies")
    lst = df3['agency'].value_counts()
    print(lst[0:20])
    print("\nBottom Agencies:")
    print(lst[20:])
    
News3K_Agency_Extract = False
if News3K_Agency_Extract:
    filename_in  = 'Final2020/Covid_8_03_news3k_all.xlsx'
    filename_out = 'Final2020/Covid_8_03_news3k_all_agency.xlsx'
    df           = pd.read_excel(filename_in)
    df = df[['agency', 'length', 'title', 'url', 'text']]
    df['source'] = "news3k"
    df['length'] = 0
    df['trump']  = 0
    df['biden']  = 0
    df['fauci']  = 0
    for i in range(df.shape[0]):
        if df['agency'].iloc[i]==  'huffington-post':
            df['agency'].iloc[i] = 'huffington'
        if df['agency'].iloc[i]==  'fox-news':
            df['agency'].iloc[i] = 'fox'
        if df['agency'].iloc[i]==  'breitbart-news':
            df['agency'].iloc[i] = 'breitbart'
        if df['agency'].iloc[i]==  'al-jazeera-english':
            df['agency'].iloc[i] = 'al-jazeera'
        if df['agency'].iloc[i]==  'nhl-news':
            df['agency'].iloc[i] = 'nhl'
        if df['agency'].iloc[i]==  'business-insider':
            df['agency'].iloc[i] = 'business'
        if df['agency'].iloc[i]==  'wall-street-journal':
            df['agency'].iloc[i] = 'business'
        if df['agency'].iloc[i]==  'fox-sports':
            df['agency'].iloc[i] = 'sports'
        if df['agency'].iloc[i]==  'nhl':
            df['agency'].iloc[i] = 'sports'
        if df['agency'].iloc[i]==  'nfl':
            df['agency'].iloc[i] = 'sports'
        if df['agency'].iloc[i]==  'bleacher-report':
            df['agency'].iloc[i] = 'sports'
        if df['agency'].iloc[i]==  'espn':
            df['agency'].iloc[i] = 'sports'
        if df['agency'].iloc[i]==  'national-review':
            df['agency'].iloc[i] = 'politics'
        if df['agency'].iloc[i]==  'hill':
            df['agency'].iloc[i] = 'politics'
        if df['agency'].iloc[i]==  'politico':
            df['agency'].iloc[i] = 'politics'
        if df['agency'].iloc[i]==  'buzzfeed':
            df['agency'].iloc[i] = 'technology'
        if df['agency'].iloc[i]==  'engadget':
            df['agency'].iloc[i] = 'technology'
        if df['agency'].iloc[i]==  'ars-technica':
            df['agency'].iloc[i] = 'technology'
        if df['agency'].iloc[i]==  'wired':
            df['agency'].iloc[i] = 'technology'
        if df['agency'].iloc[i]==  'techcrunch':
            df['agency'].iloc[i] = 'technology'
        if df['agency'].iloc[i]==  'tech-radar':
            df['agency'].iloc[i] = 'technology'
        if df['agency'].iloc[i]==  'verge':
            df['agency'].iloc[i] = 'technology'
        if df['agency'].iloc[i] not in my_agencies:
            #print(df['agency'].iloc[i])
            df['agency'].iloc[i] = "other"
        # Count number of times trump, biden and fauci appear in articles
        txt = df['text'].iloc[i].lower()
        df['length'].iloc[i] = len(txt)
        df['trump'].iloc[i] = txt.count('trump')
        df['biden'].iloc[i] = txt.count('biden')
        df['fauci'].iloc[i] = txt.count('fauci')
    print("")
    df = df[df.length>min_characters]
    for ag in np.sort(df['agency'].unique()):
        n = df.shape[0]
        t = sum(df['agency']==ag)
        r = 100*t/n
        print("{:.<18s} {:>4d} {:>3.0f}%".format(ag, t, r))
    print("{:.<18s} {:>4d} {:>3.0f}%".format("TOTAL", n, 100.0))
    print("\nSaving-->",filename_out)
    df4 = df.copy(deep=True)
    df4.to_excel(filename_out, index=False)
    print("Columns Saved: ", df.columns)
    print("Top 20 Agencies")
    lst = df4['agency'].value_counts()
    print(lst[0:20])
    print("\nBottom Agencies:")
    print(lst[20:])

Merge_Covid_Files = False
if Merge_Covid_Files:
    filename3    = 'Final2020/Covid_8_03_api_all_agency.xlsx'
    filename4    = 'Final2020/Covid_8_03_news3k_all_agency.xlsx'
    filename_out = 'Final2020/Covid_8_03_all_agency.xlsx'
    df3 = pd.read_excel(filename3)
    df4 = pd.read_excel(filename4)
    df = df3.append(df4)
    df = df.drop('title', axis=1)
    print("Combined DataFrame:", df.shape[0], "records")
    df.sort_values('url', inplace=True)
    df.drop_duplicates(subset='url', inplace=True)
    print("After dropping duplicate URLs, it has:", df.shape[0], "records")
    df.to_excel(filename_out, index=False)

Merge_Elect_Files = False
if Merge_Elect_Files:
    filename3    = 'Final2020/Nov2020_Elections_8_02_api_all_agency.xlsx'
    filename4    = 'Final2020/Nov2020_Elections_8_03_news3k_all_agency.xlsx'
    filename_out = 'Final2020/Nov2020_Elections_8_03_all_agency.xlsx'
    df3 = pd.read_excel(filename3)
    df4 = pd.read_excel(filename4)
    df  = df3.append(df4)
    df  = df.drop('title', axis=1)
    print("Combined DataFrame:", df.shape[0], "records")
    df.sort_values('url', inplace=True)
    df.drop_duplicates(subset='url', inplace=True)
    print("After dropping duplicate URLs, it has:", df.shape[0], "records")
    df.to_excel(filename_out, index=False)
    
Merge_Elect = False
if Merge_Elect: # Set the source to news3k or api depending on the 
                # intent of the merge
    filename1    = 'Final2020/Nov2020_Elections_08-01_news3k_all.xlsx'
    filename2    = 'Final2020/Nov2020_Elections_7_31_news3k_all.xlsx'
    filename_out = 'Final2020/Nov2020_Elections_08-01_news3k_all.xlsx'
    df1 = pd.read_excel(filename1)
    df2 = pd.read_excel(filename2)
    df  = df1.append(df2)
    
    dropped = df.shape[0]
    df.sort_values('text', inplace=True)
    df.drop_duplicates(subset='text', inplace=True)
    df.to_excel(filename_out, index=False)
    dropped = dropped - df.shape[0]
    print("\n {:<40s}{:>11s}".format("Filename", "Articles"))
    print("*{:-<51s}*".format("-"))
    print(" {:.<45s}{:>5d}".format(filename1[10:],df1.shape[0]))
    print(" {:.<45s}{:>5d}".format(filename2[10:],df2.shape[0]))
    print(" {:.<45s}{:>5d}".format(filename_out[10:],df.shape[0]))
    print("*{:-<51s}*".format("-"))
    print("Saved", df.shape[0], "articles and Dropped", dropped)
    
Merge_Virus = False # Used to merge a new download with the last
if Merge_Virus: # Set the source to news3k or api depending on the 
                # intent of the merge
    filename1    = 'Final2020/Nov2020_Elections_08-01_api.xlsx'
    filename2    = 'Final2020/Nov2020_Elections_7_31_api_all.xlsx'
    filename_out = 'Final2020/Nov2020_Elections_8_01_api_all.xlsx'
    df1 = pd.read_excel(filename1)
    df2 = pd.read_excel(filename2)
    df  = df1.append(df2)
    
    dropped = df.shape[0]
    df.sort_values('text', inplace=True)
    df.drop_duplicates(subset='text', inplace=True)
    df.to_excel(filename_out, index=False)
    dropped = dropped - df.shape[0]
    print("\n {:<40s}{:>11s}".format("Filename", "Articles"))
    print("*{:-<51s}*".format("-"))
    print(" {:.<45s}{:>5d}".format(filename1[10:],df1.shape[0]))
    print(" {:.<45s}{:>5d}".format(filename2[10:],df2.shape[0]))
    print(" {:.<45s}{:>5d}".format(filename_out[10:],df.shape[0]))
    print("*{:-<51s}*".format("-"))
    print("Saved", df.shape[0], "articles and Dropped", dropped)

top_agencies = False
if top_agencies:
    filename = 'Final2020/Nov2020_Elections_8_03_all_agency.xlsx'
    df = pd.read_excel(filename)
    lst = df['agency'].value_counts()
    print("Top 20 Agencies")
    print(lst[0:20])
    print("\nBottom Agencies")
    print(lst[20:])
    