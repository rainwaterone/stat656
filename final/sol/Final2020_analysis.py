#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:17:34 2020

@author: EJones
"""
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
# Classes provided from AdvancedAnalytics ver 1.25
from AdvancedAnalytics.Text import text_plot

Analysis = "Covid"
filename = "PICKLE/"+Analysis+"_08-04_df_all_pickle.pkl"
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
    
tp, bp, fp = 0, 0, 0
tn, bn, fn = 0, 0, 0
tu, bu, fu = 0, 0, 0
for i in range(df.shape[0]):
    if df['sentiment'].iloc[i]>0:
        if df['trump_1k'].iloc[i]>0:
            tp += 1
        if df['biden_1k'].iloc[i]>0:
            bp += 1
        if df['fauci_1k'].iloc[i]>0:
            fp += 1
    elif df['sentiment'].iloc[i]<0:
        if df['trump_1k'].iloc[i]>0:
            tn += 1
        if df['biden_1k'].iloc[i]>0:
            bn += 1
        if df['trump_1k'].iloc[i]>0:
            fn += 1
    else:
        if df['trump_1k'].iloc[i]>0:
            tu += 1
        if df['biden_1k'].iloc[i]>0:
            bu += 1
        if df['trump_1k'].iloc[i]>0:
            fu += 1
pt = round(100*tp/(tp+tn+tu), 1)
pb = round(100*bp/(bp+bn+bu), 1)
pf = round(100*fp/(fp+fn+fu), 1)
nt = round(100*tn/(tp+tn+tu), 1)
nb = round(100*bn/(bp+bn+bu), 1)
nf = round(100*fn/(fp+fn+fu), 1)
print("         Sentiment by Person")
print(" Name   (+/-)   Articles      Percent")
print("{:-<20s}{:->19s}".format("*", "*"))
print(" {:5s}{:>5s}{:>11d}{:>13.1f}%".format("Trump", "+", tp, pt))
print(" {:5s}{:>5s}{:>11d}{:>13.1f}%".format("Trump", "-", tn, nt))
print("\n {:5s}{:>5s}{:>11d}{:>13.1f}%".format("Biden", "+", bp, pb))
print(" {:5s}{:>5s}{:>11d}{:>13.1f}%".format("Biden", "-", bn, nb))
print("\n {:5s}{:>5s}{:>11d}{:>13.1f}%".format("Fauci", "+", fp, pf))
print(" {:5s}{:>5s}{:>11d}{:>13.1f}%".format("Fauci", "-", fn, nf))
print("{:-<20s}{:->19s}".format("*", "*"))

labels = ['BIDEN', 'TRUMP', 'FAUCI']     
p_means = [pb, pt, pf]
n_means = [nb, nt, nf]
x = np.arange(len(labels))
width = 0.35 # Bar width
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, p_means, width, label='Positive', color='mediumblue')
rects2 = ax.bar(x + width/2, n_means, width, label='Negative', color='orangered')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percent Sentiment')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0,100)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
#fig.tight_layout()
plt.show()

n_pos = []
n_neg = []
p_pos = []
p_neg = []
print("         Sentiment by Topic")
print(" Topic   (+/-)   Articles      Percent")
print("{:-<20s}{:->19s}".format("*", "*"))
for t in df['topic'].unique():
    df1 = df.loc[df['topic'] == t]
    npos  = sum(df1['sentiment']>0)
    nneg  = sum(df1['sentiment']<0)
    nneu  = sum(df1['sentiment']==0)
    pp  = round(100*npos/(npos+nneg+nneu), 1)
    pn  = round(100*nneg/(npos+nneg+nneu), 1)
    print(" {:<5d}{:>5s}{:>11d}{:>13.1f}%".format(t, "+", npos, pp))
    print(" {:<5d}{:>5s}{:>11d}{:>13.1f}%\n".format(t, "-", nneg, pn))
    n_pos.append(npos)
    p_pos.append(pp)
    n_neg.append(nneg)
    p_neg.append(pn) 
print("{:-<20s}{:->19s}".format("*", "*"))

labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]     
p_means = p_pos
n_means = p_neg
x = np.arange(len(labels))
width = 0.35 # Bar width
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, p_means, width, label='Positive', color='mediumblue')
rects2 = ax.bar(x + width/2, n_means, width, label='Negative', color='orangered')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percent Sentiment')
ax.set_xlabel('Topics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0,70)
#autolabel(rects1)
#autolabel(rects2)
#fig.tight_layout()
plt.show()

y = np.ravel(df['sentiment'])
for i in range(9):
    j = i+1
    var = 'prob'+str(j)
    x   = np.ravel(df[var])
    fig, ax = plt.subplots()
    plt.hlines(0, 0, 1.0)
    plt.xlabel("Probability for Topic "+str(j))
    plt.ylabel("Sentiment")
    plt.plot(x, y, 'o', color='tab:brown', markersize=1)

# Fit a line to x & y
df1   = df[['trump_1k', 'sentiment']].copy()
df1.sort_values('trump_1k', inplace=True)
x     = np.ravel(df1['trump_1k'])
y     = np.ravel(df1['sentiment'])
b, a  = np.polyfit(x, y, deg=1)
y_est = a + b*x

fig, ax = plt.subplots()
ax.plot(x, y_est, '-')
plt.xlabel("'Trump'/1,000 Words")
plt.ylabel("Sentiment")
plt.hlines(0, 0, 10, colors='red', linestyles='dashed', label='Neutral')
plt.plot(x, y, 'o', color='tab:brown', markersize=1)

# Fit a line to x & y
df1   = df[['biden_1k', 'sentiment']].copy()
x     = np.ravel(df1['biden_1k'])
y     = np.ravel(df1['sentiment'])

b, a  = np.polyfit(x, y, deg=1)
y_est = a + b*x

fig, ax = plt.subplots()
ax.plot(x, y_est, '-')
plt.xlabel("'Biden'/1,000 Words")
plt.ylabel("Sentiment")
plt.hlines(0, 0, 8, colors='red', linestyles='dashed', label='Neutral')
plt.plot(x, y, 'o', color='tab:brown', markersize=1)

# Fit a line to x & y
df1   = df[['fauci_1k', 'sentiment']].copy()
x     = np.ravel(df1['fauci_1k'])
y     = np.ravel(df1['sentiment'])

b, a  = np.polyfit(x, y, deg=1)
y_est = a + b*x

fig, ax = plt.subplots()
ax.plot(x, y_est, '-')
plt.xlabel("'Fauci'/1,000 Words")
plt.ylabel("Sentiment")
plt.hlines(0, 0, 20, colors='red', linestyles='dashed', label='Neutral')
plt.plot(x, y, 'o', color='tab:brown', markersize=1)
