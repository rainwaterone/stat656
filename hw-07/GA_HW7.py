#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Edward Jones

"""

from deap import creator, base, tools, algorithms

import random, sys, time, warnings
import pandas as pd
import numpy  as np
import matplotlib.pyplot               as plt
import statsmodels.api                 as sm
import statsmodels.tools.eval_measures as em
from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT
from AdvancedAnalytics.Regression          import linreg, stepwise
from math                 import log, isfinite, sqrt, pi
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics      import mean_squared_error, r2_score, log_loss
from scipy.linalg         import qr_multiply, solve_triangular
              
def rngFit(z):
    r = maxFit(z) - minFit(z)
    return round(r, 3)

def avgFit(z):
    tot = 0.0
    cnt = 0
    for i in range(len(z)):
        if isfinite(z[i][0]):
            tot += z[i][0]
            cnt += 1
    if cnt>0:
        return round(tot/cnt, 4)
    else:
        return np.nan

def maxFit(z):
    maximum = 0
    for i in range(len(z)):
        if z[i][0] > maximum:
            maximum = z[i][0]
    return maximum

def minFit(z):
    minimum = np.inf
    for i in range(len(z)):
        if z[i][0] < minimum:
            minimum = z[i][0]
    return minimum

def cvFit(z):
    avg = avgFit(z)
    std = stdFit(z)
    if isfinite(avg):
        return round(100*std/avg, 3)
    else:
        return np.nan

def logMinFit(z):
    try:
        return round(log(minFit(z)), 6)
    except:
        return np.inf
    
def logMaxFit(z):
    try:
        return round(log(maxFit(z)), 6)
    except:
        return -np.inf

def stdFit(z):
    sum1  = 0.0
    sum2  = 0.0
    cnt   = 0
    for i in range(len(z)):
        if isfinite(z[i][0]):
            sum1 += z[i][0]
            sum2 += z[i][0] * z[i][0]
            cnt += 1
    if cnt < 2:
        return np.nan
    else:
        sumsq = (sum1*sum1)/cnt
        return round(sqrt((sum2 - sumsq)/(cnt-1)), 4)
def features_min(z):
    minimum = np.inf
    feature = np.inf
    for i in range(len(z)):
        if z[i][0] < minimum:
            minimum = z[i][0]
            feature = z[i][1]
        if z[i][0] == minimum and z[i][1] < feature:
            feature = z[i][1]
    return round(feature,0)

def features_max(z):
    maximum = -np.inf
    feature =  np.inf
    for i in range(len(z)):
        if z[i][0] > maximum:
            maximum = z[i][0]
            feature = z[i][1]
        if z[i][0] == maximum and z[i][1] < feature:
            feature = z[i][1]
    return round(feature,0)

def geneticAlgorithm(X, y, n_population, n_generation, method=None,
                     reg=None, goodFit=None,  calcModel=None,
                     n_int=None, n_nom=None, n_frac=None):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    if method==None:
        method = 'random'
    if goodFit==None:
        goodFit='bic'
    if calcModel==None:
        calcModel='statsmodels'
    if type(y)==np.ndarray:
        nval = len(np.unique(y))
    else:
        nval = y.nunique()
    if reg==None:
        if nval > 20:
            reg = 'linear'
        else: 
            reg = 'logistic'
    if goodFit.lower()!='adjr2':
        opt = -1.0 # Minimize goodness of fit
    else:
        opt =  1.0 # Maximize goodness of fit
 # create individual fitness dictionary
    ifit = {}
    # create individual
    # Two weights for two optimization (goodness of fit, number of features)
    # A negative weight indicates minimize that function.
    # A positive weight indicates maximize that function.
    with warnings.catch_warnings():  
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        creator.create("FitnessMax", base.Fitness, weights=(opt, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("population_guess", initPopulation, list, 
                                                      creator.Individual)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                                     toolbox.attr_bool, n=len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
                                                      toolbox.individual)
    if   reg.lower()=='logistic':
        toolbox.register("evaluate", evalFitnessLogistic, X=X, y=y, 
                         goodFit=goodFit, calcModel=calcModel, ifit=ifit)
    elif reg.lower()=='linear':
        toolbox.register("evaluate", evalFitnessLinear, X=X, y=y, 
                         goodFit=goodFit, calcModel=calcModel, ifit=ifit)
    else:
        raise ValueError("reg not set to 'linear' or 'logistic'")
        sys.exit()
    toolbox.register("mate",     tools.cxTwoPoint)
    toolbox.register("mutate",   tools.mutFlipBit, indpb=0.02)
    toolbox.register("select",   tools.selTournament, tournsize=7)

    if method=='random':
        pop   = toolbox.population(n_population)
    else:
        # initialize parameters
        # n_int Total number of interval features
        # n_nom List of number of dummy variables for each categorical var
        pop   = toolbox.population_guess(method, n_int, n_nom, n_frac)
        #n_population = len(pop)
    hof   = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    if goodFit.lower!='adjr2':
        stats.register("features", features_min)
    else:
        stats.register("features", features_max)
    stats.register("range",    rngFit)
    stats.register("min",      minFit)
    stats.register("avg",      avgFit)
    stats.register("max",      maxFit)
    if goodFit.lower()!='adjr2':
        stats.register("Ln(Fit)",  logMinFit)
    else:
        stats.register("Ln(Fit)",  logMaxFit)
        

    # genetic algorithm
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.5,
                                   ngen=n_generation, stats=stats, 
                                   halloffame=hof, verbose=True)

    # return hall of fame
    return hof, logbook

def evalFitnessLinear(individual, X, y, goodFit, calcModel, ifit):
    # returns (goodness of fit, number of features)
    cols  = [index for index in range(len(individual)) 
            if individual[index] == 1 ]# get features subset, 
                                       # drop features with cols[i] != 1
    if type(X)==np.ndarray:
        X_selected = X[:, cols]
    else:
        X_selected = X.iloc[:, cols]
    
    features = X_selected.shape[1]
    n = X_selected.shape[0]
    p = features
    k = features + 2 # 2 for intercept and variance
    ind = ""     
    for i in range(len(individual)):
        if individual[i] == 0:
            ind += '0'
        else:
            ind += '1'
    try:
        fit = ifit[ind]
        return(fit, features)
    except:
        pass
    goodFit   = goodFit.lower()
    calcModel = calcModel.lower()
    if   k > n+2 and goodFit=='bic':
        return (np.inf, features)
    elif k > n+2 and goodFit=='adjr2':
        return (0, features)
    
    if calcModel == "qr_decomp":
        Xc     = sm.add_constant(X_selected)
        qty, r = qr_multiply(Xc, y)
        coef   = solve_triangular(r, qty)
        pred   = (Xc @ coef)
        resid  = pred - y
        ASE    = (resid @ resid) / n
        if ASE > 0:
            twoLL = n*(log(2*pi) + 1.0 + log(ASE))
            bic   = twoLL + log(n)*k
            aic   = twoLL + 2*k
            R2    = r2_score(y, pred)
            if R2 > 0.99999:
                bic = -np.inf
        else: 
            bic = -np.inf
            aic = -np.inf
            R2  = 1.0
            
        if goodFit == 'bic':
            return(bic, features)
        elif goodFit == 'aic':
            return(aic, features)
        else:
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)

    elif calcModel== "statsmodels":
        Xc       = sm.add_constant(X_selected)
        model    = sm.OLS(y, Xc)
        results  = model.fit()
        if goodFit == "adjr2":
            return(results.rsquared_adj, features)
        elif goodFit=='bic':
                bic  = em.bic(results.llf, n, k)
                return(bic, features)
        elif goodFit=='aic':
                aic  = em.aic(results.llf, n, k)
                return(aic, features)
        
    elif calcModel=='sklearn':
        # sklearn linear regression does not handle no features
        if X_selected.shape[1]>0:
            lr   = LinearRegression().fit(X_selected,y)
            pred = lr.predict(X_selected)
        else:
            avg  = y.mean()
            pred = np.array([avg]*y.shape[0])
        ASE  = mean_squared_error(y,pred)
        if ASE > 0:
            twoLL = n*(log(2*pi) + 1.0 + log(ASE))
            bic   = twoLL + log(n)*k
            aic   = twoLL + 2*k
            R2 = r2_score(y, pred)
            if R2 > 0.99999:
                bic = -np.inf
        else: 
            R2  = r2_score(y, pred)
            bic = -np.inf
            aic = -np.inf
            
        if goodFit == 'bic':
            return(bic, features)
        elif goodFit=='aic':
            return(aic, features)
        else:
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)
    else:
        raise ValueError("calcModel not 'statsmodels', 'sklearn', or 'QR_decomp'")
        sys.exit()
    
def evalFitnessLogistic(individual, X, y, goodFit, calcModel, ifit):
    # Number of categories in y
    if type(y)==np.ndarray:
        n_cat = len(np.unique(y))
    else:
        n_cat = y.nunique()
    # returns (goodness of fit, number of features)
    cols  = [index for index in range(len(individual)) 
            if individual[index] == 1 ]# get features subset, 
                                       # drop features with cols[i] != 1
    if type(X)==np.ndarray:
        X_selected = X[:, cols]
    else:
        X_selected = X.iloc[:, cols]
    
    features = X_selected.shape[1]
    n = X_selected.shape[0]
    p = features
    if n_cat <= 2:
        k = features + 2 #for intercept and varianc
    else:
        k = n_cat*(features + 1) + 1 # n_cat intercepts and +1 for variance
    ind = ""     
    for i in range(len(individual)):
        if individual[i] == 0:
            ind += '0'
        else:
            ind += '1'
    try:
        fit = ifit[ind]
        return(fit, features)
    except:
        pass
    goodFit   = goodFit.lower()
    calcModel = calcModel.lower()
    if   k > n+2 and goodFit=='bic':
        return (np.inf, features)
    elif k > n+2 and goodFit=='adjr2':
        return (0, features)

    if calcModel== "statsmodels":
        Xc = sm.add_constant(X_selected)
        try:
            model   = sm.Logit(y, Xc)
            results = model.fit(disp=False) 
        except:
            print("Singular Fit Encountered with", features, "features")
            if goodFit != 'adjr2':
                return(-np.inf, features)
            else:
                return(1.0, features)
        proba = model.predict(results.params)   
        if goodFit == "adjr2":
            R2    = r2_score(y, proba)
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)
        
        elif goodFit=='bic':
                bic  = em.bic(results.llf, n, k)
                return(bic, features)
        elif goodFit=='aic':
                aic  = em.aic(results.llf, n, k)
                return(aic, features)
        
    elif calcModel=='sklearn':
        # sklearn linear regression does not handle no features
        if X_selected.shape[1]>0:
            if X_selected.shape[0]*X_selected.shape[1] > 100000:
                opt='saga'
            else:
                opt='lbfgs'
            lr    = LogisticRegression(penalty='none', solver=opt,
                                  tol=1e-4, max_iter=5000)
            lr    = lr.fit(X_selected, y)
            proba = lr.predict_proba(X_selected)
        else:
            proba = np.full((y.shape[0],2),0.5)
        
        ll = -log_loss(y, proba[:, 1]) * n
        twoLL = -2.0*ll
        if goodFit == 'bic':
            bic   = twoLL + log(n)*k
            return(bic, features)
        elif goodFit=='aic':
            aic   = twoLL + 2*k
            return(aic, features)
        else:
            R2 = r2_score(y, proba[:,1])
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)
    else:
        raise ValueError("calcModel not 'statsmodels', 'sklearn', or 'QR_decomp'")
        sys.exit()
    
def initPopulation(pcls, ind_init, method, 
                   n_int, n_nom, n_frac):
    #k = number of columns in X
    #k1= number of interval variables (first k1 columns)
    #k2= number of other columns in X
    if n_int==None:
        k1 = 0
    elif type(n_int)==int:
        k1 = n_int
    else:
        k1 = 0
        
    if n_nom==None:
        k2 = 0
    elif type(n_nom)==int:
        k2 = n_nom
    else:
        k2 = sum(n_nom)
    k = k1+k2
    # Initialize Null Case (no features)
    icls = [0]*k
    ind  = ind_init(icls)
    pcls = [ind]
    
    if method == 'star':
        # Add "All" one-feature selection (star points)
        for i in range(k):
            icls = [0]*k
            icls[i]  = 1
            ind = ind_init(icls)
            pcls.append(ind)
            
    return pcls

def findBest(hof, goodFit, X, y, top=None):
    #Find Best Individual in Hall of Fame
    print("Individuals in HoF: ", len(hof))
    if top==None:
        top=1
    goodFit = goodFit.lower()
    features = np.inf
    if goodFit=='bic' or goodFit=='aic':
        bestFit = np.inf
        for individual in hof:
            if(individual.fitness.values[0] < bestFit):
                bestFit = individual.fitness.values[0]
                _individual = individual
            if (sum(individual) < features and 
                individual.fitness.values[0] == bestFit):
                features = sum(individual)
                _individual = individual
    elif goodFit=='adjr2':
        bestFit = -np.inf
        for individual in hof:
            if(individual.fitness.values[0] > bestFit):
                bestFit = individual.fitness.values[0]
                _individual = individual
            if (sum(individual) < features and 
                individual.fitness.values[0] == bestFit):
                features = sum(individual)
                _individual = individual
    else:
        raise RuntimeError("goodFit invalid: "+goodFit)
        sys.exit()
    if type(X)==np.ndarray:
        z = np.ravel(_individual)
        z = z.nonzero()
        _individualHeader = z[0]
    else:
        _individualHeader = [list(X)[i] for i in range(len(_individual)) 
                        if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader

        
def plotGenerations(gen, lnbic, features):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("GA GENERATION", fontsize="x-large",fontweight="heavy")
    ax1.tick_params(axis='x', labelcolor="black", labelsize="x-large")
    ax1.tick_params(axis='y', labelcolor="green", labelsize="x-large")
    #ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax1.set_ylabel("Log(BIC)", fontsize="x-large", fontweight="heavy", 
                   color="green")
    ax1.set_facecolor((0.95,0.95,0.95))
    #ax1.grid(axis='x', linestyle='--', linewidth=1, color='gray')
    ax1.plot(gen, lnbic, 'go-', color="green", 
                         linewidth=2, markersize=10)
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor="blue", labelsize="x-large")
    ax2.set_ylabel("Number of Features Selected", fontsize="x-large", 
                   fontweight="heavy", color="blue")
    #ax2.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    #ax2.grid(axis='y', linestyle='--', linewidth=1, color='gray')
    ax2.plot(gen, features, 'bs-', color="blue", 
                         linewidth=2, markersize=10)
    plt.savefig("GA_Feature_Select.pdf")
    plt.show()
#*****************************************************************************
print("{:*>71s}".format('*'))
attribute_map = {
    'obs':      [DT.Ignore,  (1, 53940)],
    'price':    [DT.Interval,(300, 20000)],
    'carat':    [DT.Interval,(0.2, 5.5)],
    'cut':      [DT.Nominal,('Fair', 'Good', 'Ideal', 'Premium', 
                             'Very Good')],
    'color':    [DT.Nominal,('D', 'E', 'F', 'G', 'H', 'I', 'J')],
    'clarity':  [DT.Nominal,('I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 
                             'VVS1', 'VVS2')],
    'depth':    [DT.Interval,(40, 80)], 
    'table':    [DT.Interval,(40, 100)],
    'x':        [DT.Interval,(0, 11)],
    'y':        [DT.Interval,(0, 60)],
    'z':        [DT.Interval,(0, 32)] }
    
target = "price"
df  = pd.read_excel("diamonds_train.xlsx")
dfv = pd.read_excel("diamonds_validation.xlsx")
print("Read", df.shape[0], " training observations with ", 
      df.shape[1], "attributes:\n")
print("Read", dfv.shape[0], " validation observations with ", 
      dfv.shape[1], "attributes:\n")

# Note seems you can use drop=False or True, but True works better and avoid
# possible runtime errors

rie = ReplaceImputeEncode(data_map=attribute_map,
                          nominal_encoding='one-hot',
                          drop=False, display=True)
encoded_df  = rie.fit_transform(df)
encoded_dfv = rie.fit_transform(dfv)

y = encoded_df[target] # The target is not scaled or imputed
X = encoded_df.drop(target, axis=1)

yv = encoded_dfv[target] # The target is not scaled or imputed
Xv = encoded_dfv.drop(target, axis=1)

print("{:*>71s}".format('*'))
# apply genetic algorithm
# n_init:  set to the number of candidate interval and binary features
# n_nom:   set to a list of levels for each candidate nominal feature
#          if there are no candidate nominal features, set to an empty list []
n_int =  6       # an integer 0 or greater
n_nom = [5, 7, 8] # 20 dummy features before drop, 17 after drop
#n_nom = [4, 6, 7] # 17 dummy features if drop=True
p     = n_int + sum(n_nom) # Total number of features 52

# modes:   the list of currently available statistical models
# fitness: the list of currently available fitness functions
# init:    the list of currently available initialization algorithms
#          each initialization algorithm can be used to initialize 
#          generation zero.  Select the one that produces a generation zero
#          closest to the imagined best number of features.  'star' starts 
#          with only one feature per individual.  'random' starts with a
#          larger number of features per individual, approximate half the
#          total number of candidates.
models     = [ 'sklearn', 'statsmodels', 'QR_decomp']
fitness    = ['bic', 'aic', 'AdjR2']
init       = ['star', 'random']
# Set calcModel, goodFit and initMethod to your choice for the statistical
#     model, the goodness of fit metric, and the initialization algorithm.
calcModel  = models [1]
goodFit    = fitness[0]
initMethod = init[0] #Initial generation has only 1 feature per individual.
             #Initial generation with 'random' has about 50% of all features.
# n_pop is the initial population size.  Subsequent generations will be near
#       this size.
# n_gen is the number of generations, each progressively better than the 
#       previous generation.  This needs to be large enough to all the 
#       search algorithm to identify the best feature selection.
# Note: This algorithm optimizes the fitness of the individual while 
#       minimizing the number of features selected for the model.
if initMethod=='star':
    n_pop = p+1
    n_gen =  50
else:
    n_pop = 100
    n_gen =  50

print("{:*>71s}".format('*'))
print("{:*>14s}     GA Selection using {:>5s} Fitness         {:*>11s}". 
      format('*', goodFit, '*'))
print("{:*>14s} {:>11s} Models and {:>6s} Initialization {:*>11s}". 
      format('*', calcModel, initMethod, '*'))
print(" ")
random.seed(12345)
start = time.time()
hof, logbook = geneticAlgorithm(X, y, n_pop, n_gen, method=initMethod,
                                reg='linear', goodFit=goodFit,
                                calcModel=calcModel, n_int=n_int, n_nom=n_nom)

gen, features, min_, avg_, max_, rng_, lnfit = logbook.select("gen",
                    "features", "min", "avg", "max", "range", "Ln(Fit)")
end = time.time()    
duration = end-start
print("GA Runtime ", duration, " sec.")

# Plot Fitness and Number of Features versus Generation
plotGenerations(gen, lnfit, features)

# select the best individual
fit, individual, header = findBest(hof, goodFit, X, y)
print("Best Fitness:", fit[0])
print("Number of Features Selected: ", len(header))
print("\nFeatures:", header)

# Analysis and Evaluation using StatsModels
Xc      = sm.add_constant(X[header])
model   = sm.OLS(y, Xc)
results = model.fit()
pred    = model.predict(results.params)
n = Xc.shape[0]
k = Xc.shape[1] + 1
aic      = em.aic(results.llf, n, k)
bic      = em.bic(results.llf, n, k)
ASE      = mean_squared_error(y,pred)
Root_ASE = sqrt(ASE)
print(results.summary())
print("\nTraining Data Metrics")
print("{:.<25s}{:12.4f}".format('ASE', ASE))
print("{:.<25s}{:12.4f}".format('Square Root of ASE', Root_ASE))
print("{:.<25s}{:12.4f}".format('AIC', aic))
print("{:.<25s}{:12.4f}".format('BIC', bic))
print("{:.<25s}{:12.4f}".format('Adj. R-Squared', results.rsquared_adj))

Xcv   = sm.add_constant(Xv[header])
predv = model.predict(results.params, Xcv)
ASE   = mean_squared_error(yv,predv)
Root_ASE = sqrt(ASE)
print("\nValidation Data Metrics")
print("{:.<25s}{:12.4f}".format('ASE', ASE))
print("{:.<25s}{:12.4f}".format('Square Root of ASE', Root_ASE))

# Analysis and Evalution using sklearn
lr    = LinearRegression()
lr    = lr.fit(X[header], y)
linreg.display_split_metrics(lr, X[header], y, Xv[header], yv)

rie = ReplaceImputeEncode(data_map=attribute_map, 
                          nominal_encoding="one-hot", 
                          drop=True, display=False)
encoded_df  = rie.fit_transform(df)
encoded_dfv = rie.fit_transform(dfv)

y  = encoded_df[target]
X  = encoded_df.drop(target, axis=1)
yv = encoded_dfv[target]
Xv = encoded_dfv.drop(target, axis=1)
                      
print(" ")
print("{:*>71s}".format('*'))
print("{:*>14s}     STEPWISE SELECTION    {:*>30s}". format('*', '*'))
print("{:*>71s}".format('*'))

sw       = stepwise(encoded_df, target, reg="linear", method="stepwise",
                    crit_in=0.1, crit_out=0.1, verbose=True)
selected = sw.fit_transform()
print("Number of Selected Features: ", len(selected))

# Analysis and Evalution using sklearn
lr    = LinearRegression()
lr    = lr.fit(X[selected], y)
linreg.display_split_metrics(lr, X[selected], y, Xv[selected], yv)


print(" ")
print("{:*>71s}".format('*'))
print("{:*>14s}     FIT FULL MODEL        {:*>30s}". format('*', '*'))
print("{:*>71s}".format('*'))
# Analysis and Evalution using sklearn
lr    = LinearRegression()
lr    = lr.fit(X, y)
linreg.display_split_metrics(lr, X, y, Xv, yv)

print("{:*>71s}".format('*'))
print("{:*>18s}        LASSO       {:*>33s}". format('*', '*'))
print("{:*>71s}".format('*'))

alpha_list = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
alpha_lasso = np.array(alpha_list)
col = X.columns
best_vase = np.inf
vase_     = []
m         = 0
coefs_lasso = np.zeros((X.shape[1], len(alpha_list)))
for a in alpha_list:
    clr = Lasso(alpha=a, tol=1e-8, random_state=12345)
    clr.fit(X, y)
    coef = clr.coef_        
    coefs_lasso[:, m] = coef
    m += 1
    features = []
    for i in range(len(coef)):
        if coef[i] != 0:
            features.append(col[i])
    if a == 4:
        print(features)
    print("\nAlpha: ", a, " Number of Coefficients: ", len(features), "/", len(coef))
    if len(features) == 0:
        print("alpha=", a, " all coefficients are zero")
        continue
    clr.fit(X[features], y)
    pred = clr.predict(Xv[features])
    vase = mean_squared_error(yv, pred)
    vase_.append(vase)
    if vase < best_vase:
        best_vase  = vase
        best_alpha = a
    linreg.display_split_metrics(clr, X[features], y, Xv[features], yv)
"""
print("\nThe Alpha that minimized VASE: ", best_alpha)
# Scale coefficients as percent change from base : smallest alpha
for i in range(len(alpha_list)-1):
    j = i+1
    for k in range(X.shape[1]):
        coef[k] = (coefs_lasso[k,j])/coefs_lasso[k,0]
    coefs_lasso[:, j] = coef
coefs_lasso[:, 0] = np.zeros(X.shape[1])
"""
from itertools import cycle
plt.figure(1)
colors = cycle(['b', 'r', 'g', 'c', 'k'])
for coef_l,  c in zip(coefs_lasso[0:6,:], colors):
    l1 = plt.plot(alpha_lasso, coef_l, c=c, linestyle='dotted', linewidth=2)
plt.hlines(0, 0, 10, linestyles='dashdot', colors='b', linewidth=1)
plt.vlines(4, -1000, 11100, linestyles='dashed', colors='k', linewidth=1)
plt.xlabel('alpha')
plt.xlabel('alpha')
plt.ylabel('Interval Coefficients')
plt.title('Interval Coefficients vs Alpha')
plt.axis('tight')

plt.figure(2)
colors = cycle(['b', 'r', 'g', 'c', 'k'])
for coef_l,  c in zip(coefs_lasso[6:10,:], colors):
    l1 = plt.plot(alpha_lasso, coef_l, c=c, linestyle='dotted', linewidth=2)
plt.hlines(0, 0, 10, linestyles='dashdot', colors='b', linewidth=1)
plt.vlines(4, -800, 200, linestyles='dashed', colors='k', linewidth=1)
plt.xlabel('alpha')
plt.xlabel('alpha')
plt.ylabel('Cut Coefficients')
plt.title('Cut Coefficients vs Alpha')
plt.axis('tight')

plt.figure(3)
colors = cycle(['b', 'r', 'g', 'c', 'k'])
for coef_l,  c in zip(coefs_lasso[10:16,:], colors):
    l1 = plt.plot(alpha_lasso, coef_l, c=c, linestyle='dotted', linewidth=2)
plt.hlines(0, 0, 10, linestyles='dashdot', colors='b', linewidth=1)
plt.vlines(4, -100, 2500, linestyles='dashed', colors='k', linewidth=1)
plt.xlabel('alpha')
plt.xlabel('alpha')
plt.ylabel('Color Coefficients')
plt.title('Color Coefficients vs Alpha')
plt.axis('tight')

plt.figure(4)
colors = cycle(['b', 'r', 'g', 'c', 'k'])
for coef_l,  c in zip(coefs_lasso[16:23,:], colors):
    l1 = plt.plot(alpha_lasso, coef_l, c=c, linestyle='dotted', linewidth=2)
plt.hlines(0, 0, 10, linestyles='dashdot', colors='b', linewidth=1)
plt.vlines(4, -5500, 1000, linestyles='dashed', colors='k', linewidth=1)
plt.xlabel('alpha')
plt.ylabel('Clarity Coefficients')
plt.title('Clarity Coefficients vs Alpha')
plt.axis('tight')