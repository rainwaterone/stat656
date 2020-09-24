"""
Created 09 JUN 2020

@author: el-rainwater, Rainwater Center for Neolithic Computing
"""

import pandas as pd
import numpy as np
from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT
from AdvancedAnalytics.Regression import logreg, stepwise
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

filepath = '/Users/edwardrainwater/OneDrive - Texas A&M University/' \
    'Summer-2020/STAT 656 Applied Analytics/hw-03/'
file = 'credithistory.xlsx'

def run_all_features(encoded_df,target):
    print("\n" + ("*"*78))
    print("*"*13 + "  Running StatsModel - All Features with Validation   " + "*"*13)
    print(("*"*78))
    y = encoded_df[target].astype(int) 
    X = encoded_df.drop(target, axis=1)
    Xt, Xv, yt, yv = train_test_split(X, y, train_size=0.7, random_state=12345)
    Xtc     = sm.add_constant(Xt)
    Xvc     = sm.add_constant(Xv)
    model   = sm.Logit(yt, Xtc)
    results = model.fit()
    print('Printing results summary...')
    print(results.summary())
    
    print("****** Training Model Metrics *****")
    mat = results.pred_table(threshold=0.5)
    logreg.display_confusion(mat)
    
    print("\n******** Validation Metrics *******")
    predv  = results.predict(Xvc)
    sv     = np.where(predv<0.5, 0, 1)
    logreg.display_confusion(pd.crosstab(sv, yv))
    print("**********************************************\n")
    return()


def run_stepwise(df_encoded,target):
    print("\n" + ("*"*78))
    print("*"*15 + "  Running StatsModel - Logistic with Stepwise   " + "*"*15)
    print(("*"*78))
    
    # Set up stepwise feature selection
    df_encoded[target] = df_encoded[target].astype(int) # Do this to make target int
    y = df_encoded[target]
    sw = stepwise(df_encoded, target, reg='logistic', verbose=True)
    
    selected = sw.fit_transform()
    
    print('\nFinal selected attributes:')
    print(*selected,sep='\n')
    print(("*"*78))
    
    # Split the model 70/30 for training/validation
    
    X_train, X_validate, y_train, y_validate =  \
        train_test_split(df_encoded[selected], y, train_size=0.7, 
                         random_state = 12345)
    
    Xc_train    = sm.add_constant(X_train)
    Xc_validate = sm.add_constant(X_validate)
    
    model    = sm.Logit(y_train, Xc_train)
    results  = model.fit()
    print("\n" + ("*"*78))
    print("*"*31 + " Training Model " + "*"*31)
    print("                              Target: " + target)
    print(results.summary())
    
    print("\n" + ("*"*78))
    print("*"*27 + " Training Model Metrics " + "*"*27)
    print(("*"*80))

    mat = results.pred_table(threshold=0.5)
    logreg.display_confusion(mat)
    
    print("\n" + ("*"*78))
    print("*"*29 + " Validation Metrics " + "*"*29)
    print(("*"*78))

    predv  = results.predict(Xc_validate)
    sv     = np.where(predv<0.5, 0, 1)
    logreg.display_confusion(pd.crosstab(sv, y_validate))
    print(("*"*78))

    return()

def run_reglr_logistic_regression(encoded_df, target):
    print("*******Regularization Logistic Regression*****")
    y = encoded_df[target].astype(int) 
    X = encoded_df.drop(target, axis=1)
    X_train, X_validate, y_train, y_validate =  \
    train_test_split(X, y, train_size=0.7, random_state = 12345)

    C_list = [1e-4, 1e-2, 1e-1, 1.0, 5.0, 10.0, 50.0, np.inf]
    for c in C_list:    
        lr = LogisticRegression(C=c, tol=1e-4, solver='lbfgs', max_iter=5000)
        lr = lr.fit(X_train, y_train)
        print("\nLogistic Regression Model using C=", c)
        logreg.display_split_metrics(lr, X_train, y_train, X_validate, 
                                     y_validate, target_names=['Bad', 'Good'])
        
    print("\n** Cross-Validation for Regularization Logistic Regression **")
    for c in C_list:
        lr = LogisticRegression(C=c, tol=1e-4, solver='lbfgs', max_iter=5000)
        lrc = cross_val_score(lr, X, y, cv=10, scoring='f1', n_jobs=3)
        mean = lrc.mean()
        std  = lrc.std()
        print("{:.<10.6f}{:>10.4f}{:>10.4f}".format(c, mean, std))
    return()



def main(): ##########################################################

    df = pd.read_excel(filepath + file)
    print(df.head(), df.shape)
    print(df.dtypes)
    
    attribute_map = {
        'age':[DT.Interval, (19,120)],
        'amount':[DT.interval, (0,20000)],
        'checking':[DT.nominal, (list(range(1,5)))],
        'coapp':[DT.nominal, (1,2,3)],
        'depends':[DT.Binary, (1,2)],
        'duration':[DT.interval, (1,72)],
        'employed':[DT.Nominal, (list(range(1,6)))],
        'existcr':[DT.Nominal, (1,2,3,4)],
        'foreign':[DT.Binary, (1,2)],
        'history':[DT.nominal, (0,1,2,3,4)],
        'housing':[DT.Nominal, (1,2,3)],
        'installp':[DT.nominal, (1,2,3,4)],
        'job':[DT.nominal, (1,2,3,4)],
        'marital':[DT.Nominal, (1,2,3,4)],
        'other':[DT.Nominal, (1,2,3)],
        'property':[DT.nominal, (1,2,3,4)],
     	'purpose':[DT.Ignore, ('0', '1', '2', '3', '4', '5', \
                              '6', '8', '9', 'X') ],
        'resident':[DT.Nominal, (1,2,3,4)],
        'savings':[DT.nominal,(1,2,3,4,5)],
        'telephon':[DT.Binary, (1,2)],
       	'good_bad':[DT.Binary , ('bad', 'good') ]
        }
    
    
    target = 'good_bad'
    
    # One-hot encode and impute missing values
    rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot',
                              binary_encoding='one-hot', no_impute=[target], 
                              interval_scale=None, drop=True,
                              display=True)
    df_encoded = rie.fit_transform(df).dropna() #drop rows with missing values
    
    run_all_features(df_encoded,target)
    run_stepwise(df_encoded,target)
    run_reglr_logistic_regression(df_encoded, target)



if __name__=='__main__':
    main()