"""
Created 09 JUN 2020

@author: el-rainwater, Rainwater Center for Neolithic Computing

STAT 656, Homework No. 4
"""
import time
start_time = time.time()
import pandas as pd
import numpy as np
from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT
from AdvancedAnalytics.Regression import logreg
from AdvancedAnalytics.Tree import tree_classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
import pickle
from copy import deepcopy

# Use the path below on MacOS
# filepath = '/Users/edwardrainwater/OneDrive - Texas A&M University/' \
#     'Summer-2020/STAT 656 Applied Analytics/hw-04/'

filepath = 'C:/Users/rainwater-e/OneDrive - Texas A&M University/' \
    'Summer-2020/STAT 656 Applied Analytics/hw-04/'
file = 'CellphoneActivity.xlsx'

df = pd.read_excel(filepath + file)
print(df.head(), df.shape)
print(df.dtypes)

target_categories = ['walking', 'standingup', 'standing', 'sittingdown',
                     'sitting']

attribute_map = {
    'user':[DT.Nominal, ('wallace', 'katia', 'jose_carlos', 'debora')],
    'gender':[DT.binary, ('Man', 'Woman')],
    'age':[DT.interval, (20,80)],
    'height':[DT.interval, (1.5,1.75)],
    'weight':[DT.interval, (50,85)],
    'BMI':[DT.interval, (20,30)],
    'x1':[DT.interval,(-750,750)],
    'y1':[DT.interval,(-750,750)],
    'z1':[DT.interval,(-750,750)],
    'x2':[DT.interval,(-750,750)],
    'y2':[DT.interval,(-750,750)],
    'z2':[DT.interval,(-750,750)],
    'x3':[DT.interval,(-750,750)],
    'y3':[DT.interval,(-750,750)],
    'z3':[DT.interval,(-750,750)],
    'x4':[DT.interval,(-750,750)],
    'y4':[DT.interval,(-750,750)],
    'z4':[DT.interval,(-750,750)],
   	'activity':[DT.ignore,(target_categories)]
    }
    
    # 'employed':[DT.Nominal, (list(range(1,6)))],



target = 'activity'

# One-hot encode and impute missing values
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot',
                          binary_encoding='one-hot', 
                          interval_scale=None, drop=True,
                          display=True)
encoded_df = rie.fit_transform(df).dropna() #drop rows with missing values


print("******* Regularization Logistic Regression *****")
# print("********* Setting up Cross-Validation **********")


# y = encoded_df[target].astype(int)
# y = encoded_df[target]
y = df[target]
# X = encoded_df.drop(target, axis=1)
X = encoded_df # because 'activity' was ignored in RIE, we don't have to drop it

X_train, X_validate, y_train, y_validate =  \
train_test_split(X, y, train_size=0.7, random_state = 12345)

C_list = [1e-4, 1e-2, 1e-1, 1.0, 5.0, 10.0, 50.0, np.inf]
score_list = ['precision_macro', 'recall_macro', 'f1_macro']
best_f1_score = 0
# for c in C_list:    
#     lr = LogisticRegression(C=c, tol=1e-4, solver='lbfgs', max_iter=10000)
#     lr = lr.fit(X_train, y_train)
#     print("\nLogistic Regression Model using C=", c)
#     logreg.display_split_metrics(lr, X_train, y_train, X_validate, 
#                                  y_validate, target_names=target_categories)
    
print('\n** Cross-Validation for Regularization Logistic Regression **')


for c in C_list:
    print('\nPerforming Logistic Regression for C = ', c)
    print('{:.<18s}{:>6s}{:>13s}'.format('Metric', 'Mean', 'Std. Dev.'))
    lr = LogisticRegression(C=c, tol=1e-4, solver='lbfgs', max_iter=1000)
    lrc = cross_validate(lr, X, y, cv=10, scoring=score_list,
                          return_train_score=False, n_jobs=-1)
    
    for s in score_list:
        var  = 'test_'+s
        mean = lrc[var].mean()
        std  = lrc[var].std()
        print('{:.<18s}{:>7.4f}{:>10.4f}'.format(s, mean, std))
        if s == 'f1_macro' and mean > best_f1_score:
            best_f1_score = mean
            best_c  = c
            
# Set regression to be of the best regularization parameter
lr = LogisticRegression(C=best_c, tol=1e-4, solver='lbfgs', max_iter=1000)
lrfit = lr.fit(X,y)
print('\nLogistic Parameter of Best C = ', best_c)
logreg.display_metrics(lr, X, y)

# Run 70/30 Validation of Logistic Regression
Xt, Xv, yt, yv = train_test_split(X, y, test_size = 0.3, random_state=12345)
lr = LogisticRegression(C = best_c, tol=1e-4, solver='lbfgs', max_iter=1000)
lrfit = lr.fit(X,y)
logreg.display_split_metrics(lr, Xt, yt, Xv, yv)



# Decision Tree with 10-fold Cross-validation

y = df[target]
X = rie.fit_transform(df)
# np_y = np.ravel(y) # Ravel it into a contiguous flattened array
best = 0

depths = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25]

for d in depths:
    print("\nTree Depth: ", d)
    dtc = DecisionTreeClassifier(max_depth=d, 
                                 min_samples_leaf=5, 
                                 min_samples_split=5,
                                 random_state=12345)
    dtc = dtc.fit(X,y)
    scores = cross_validate(dtc, X, y, scoring=score_list, 
                            return_train_score=False, cv=10)
    
    print("{:.<18s}{:>6s}{:>13s}".format("Metric", "Mean", 
                                          "Std. Dev."))
    for s in score_list:
        var = "test_"+s
        mean = scores[var].mean()
        std  = scores[var].std()
        print("{:.<18s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        if s=='f1_macro' and mean>best:
            best = mean
            best_depth = d
            best_tree = deepcopy(dtc) # copies dtc and all nested objects


print("\nBest Tree Depth: ", best_depth)
tree_classifier.display_importance(best_tree, X.columns.values, 
                                top=15, plot=True)
tree_classifier.display_metrics(best_tree, X, y)

# Pickle Best Decision Tree Model
pickle.dump(best_tree, open('BestTree_All.pkl', 'wb'))

print("************ Decision Tree 70/30 Validation ************")
Xt, Xv, yt, yv = \
    train_test_split(X,y,test_size = 0.3, random_state=12345)

dtc = DecisionTreeClassifier(max_depth=best_depth, \
                             min_samples_leaf=5,   \
                             min_samples_split=5,  \
                             random_state=12345)
dtc.fit(Xt, yt)
tree_classifier.display_importance(dtc, Xt.columns.values, 
                                top=15, plot=True)
tree_classifier.display_split_metrics(dtc, Xt, yt, Xv, yv)

# Pickle Best Decision Tree Model
pickle.dump(dtc, open('BestTree_Train.pkl', 'wb'))

print('\nExecution Time: ',round(time.time()-start_time,2),' seconds')