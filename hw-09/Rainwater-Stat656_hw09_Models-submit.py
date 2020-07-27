"""
STAT 656 HW-09
@author:Lee Rainwater
@heavy_lifting_by: Dr. Edward Jones
@date: 2020-07-23

"""

import sys, warnings, random, time
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
# Classes provided for the course
from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT
from AdvancedAnalytics.Regression          import logreg, stepwise
from AdvancedAnalytics.Tree                import tree_classifier
from AdvancedAnalytics.Forest              import forest_classifier

import statsmodels.api                 as sm
import statsmodels.tools.eval_measures as em

from math                    import log, isfinite, sqrt, pi
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import VotingClassifier
from sklearn.metrics         import mean_squared_error, r2_score, log_loss
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import cross_val_score
from scipy.linalg            import qr_multiply, solve_triangular
from deap                    import creator, base, tools, algorithms
from copy                    import deepcopy

     
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
            lr   = LogisticRegression(penalty='none').fit(X_selected,y)
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
                return(np.inf, features)
            else:
                return(0.0, features)
        
        if goodFit == "adjr2":
            proba = model.predict(results.params)
            R2    = r2_score(y, proba)
            adjr2 = 1.0-R2 
            adjr2 = ((n-1)/(n-p-1))*adjr2
            if adjr2 < 1.0:
                adjr2 = 1.0 - adjr2
                return(adjr2, features)
            else:
                return(0.0, features)
        elif goodFit=='bic':
                bic = em.bic(results.llf, n, k)
                return(bic, features)
        elif goodFit=='aic':            
                aic = em.aic(results.llf, n, k)
                return(aic, features)
        else:
            raise RuntimeError("Goodness of Fit not AdjR2, BIC or AIC")
        
    elif calcModel=='sklearn':
        # sklearn linear regression does not handle logistic with no features
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
    ax1.set_xlabel("GA Generation", fontsize="x-large",fontweight="heavy")
    ax1.tick_params(axis='x', labelcolor="black", labelsize="x-large")
    ax1.tick_params(axis='y', labelcolor="green", labelsize="x-large")
    #ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax1.set_ylabel("Log(Fitness)", fontsize="x-large", fontweight="heavy", 
                   color="green")
    ax1.set_facecolor((0.95,0.95,0.95))
    #ax1.grid(axis='x', linestyle='--', linewidth=1, color='gray')
    ax1.plot(gen, lnbic, 'go-', color="green", 
                         linewidth=2, markersize=10)
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor="blue", labelsize="x-large")
    ax2.set_ylabel("Features Selected (n)", fontsize="x-large", 
                   fontweight="heavy", color="blue")
    #ax2.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    #ax2.grid(axis='y', linestyle='--', linewidth=1, color='gray')
    ax2.plot(gen, features, 'bs-', color="blue", 
                         linewidth=2, markersize=10)
    plt.savefig("GA_Feature_Select.pdf")
    plt.show()
    
def r_plot(lr, alphas, best, X, y):
    import matplotlib.pyplot as plt
    coefs  = []
    ase    = []
    for a in alphas:
        lr.set_params(alpha=a)
        lr.fit(X, y)
        coefs.append(lr.coef_)
        predictions = lr.predict(X)
        mse = mean_squared_error(y, predictions)
        ase.append(mse)  
        
    plt.figure(figsize=(10, 3))
    plt.subplot(121)
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Regression Coefficients')
    plt.title('Coefficients vs. Shrinkage')
    plt.axis('tight')
    plt.axvline(best, color="r")
    
    plt.subplot(122)
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.plot(alphas, ase)
    ax.set_xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Avg. Squared Error')
    plt.title('ASE vs. Shrinkage')
    plt.axis('tight')
    plt.axvline(best, color="r")
    plt.savefig('Lasso.eps', pad_inches=0.1, dpi=256, bbox_inches='tight')
    plt.savefig('Lasso.png', pad_inches=0.1, dpi=256, bbox_inches='tight')
    plt.show()
    
def e_plot(lr, alphas, l1_ratios, best, X, y):
    import matplotlib.pyplot as plt
    for l1 in l1_ratios:
        coefs  = []
        ase    = []
        for a in alphas:
            lr.set_params(alpha=a, l1_ratio=l1)
            lr.fit(X, y)
            coefs.append(lr.coef_)
            predictions = lr.predict(X)
            mse = mean_squared_error(y, predictions)
            ase.append(mse)  
        
        plt.figure(figsize=(10, 3))
        plt.subplot(121)
        ax = plt.gca()
        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        plt.xlabel('Alpha')
        plt.ylabel('Regression Coefficients')
        plt.title('Coefficients vs. Shrinkage for L1='+str(l1))
        plt.axvline(best, color="r")
        plt.axis('tight')
    
        plt.subplot(122)
        ax = plt.gca()
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.plot(alphas, ase)
        ax.set_xscale('log')
        plt.xlabel('Alpha')
        plt.ylabel('Avg. Squared Error')
        plt.title('ASE vs. Shrinkage for L1=' + str(l1))
        plt.axvline(best, color="r")
        plt.axis('tight')
        plt.show() 

#*****************************************************************************
attribute_map = {
        'NhtsaID':[DT.ID, ('')], 
        'Year':    [DT.Nominal, (2001, 2002, 2003)],
        'Make':    [DT.Nominal, ('HONDA', 'ACURA')],
        'Model':   [DT.Nominal, ('TL', 'ODYSSEY', 'CR-V', 'CL', 'CIVIC', 
                                 'ACCORD')],
        'State':   [DT.Ignore],
        'description': [DT.Text,('')],
        'crash':   [DT.Binary,  ('N', 'Y')],
        'mph':     [DT.Interval,(0.0, 100.0)],
        'abs':     [DT.Binary,  ('N', 'Y')],
        'mileage': [DT.Interval,(0.0, 200000.0)],
        'cruise':  [DT.Binary,  ('N', 'Y')],
        'prob1':   [DT.Interval,(0.0,1.0)],
        'prob2':   [DT.Interval,(0.0,1.0)],
        'prob3':   [DT.Interval,(0.0,1.0)],
        'prob4':   [DT.Interval,(0.0,1.0)],
        'prob5':   [DT.Interval,(0.0,1.0)],
        'prob6':   [DT.Interval,(0.0,1.0)],
        'prob7':   [DT.Interval,(0.0,1.0)],
        'prob8':   [DT.Interval,(0.0,1.0)],
}
target = 'crash'
df     = pd.read_pickle("Honda_df_pickle.pkl")
rie    = ReplaceImputeEncode(data_map=attribute_map, no_impute=[target],
                             binary_encoding='one-hot',
                             nominal_encoding='one-hot', drop=False, 
                             display=True)
encoded_df = rie.fit_transform(df)
y = encoded_df[target] # The target is not scaled or imputed
X = encoded_df.drop(target, axis=1)

n_int = 12        # interval variable - points, excluding the target
n_nom = [3, 2, 6] # two nominal variables (Region, and topic)

# GA Selection Parameters
p     = n_int + sum(n_nom) # Total number of features 37
models     = [ 'sklearn', 'statsmodels', 'QR_decomp']
fitness    = ['bic', 'aic', 'AdjR2']
init       = ['star', 'random']
calcModel  = models [0]
goodFit    = fitness[0]
initMethod = init[0] #Initial generation has only 1 feature per individual.

if initMethod=='star':
    n_pop = p+1
    n_gen =  25
else:
    n_pop =  50
    n_gen =  25

print("\n{:-<40s}{:->39s}".format('*', '*'))
print("{:-<14s}    GA Selection using {:>5s} Fitness           {:->18s}". 
      format('*', goodFit, '*'))
print("{:-<14s}{:>11s} Models and {:>6s} Initialization   {:->18s}\n". 
      format('*', calcModel, initMethod, '*'))
print(" ")
random.seed(12345)
start = time.time()
hof, logbook = geneticAlgorithm(X, y, n_pop, n_gen, method=initMethod,
                                reg='logistic', goodFit=goodFit,
                                calcModel=calcModel, n_int=n_int, n_nom=n_nom)

gen, features, min_, avg_, max_, rng_, lnfit = logbook.select("gen",
                    "features", "min", "avg", "max", "range", "Ln(Fit)")
end = time.time()    
duration = end-start
print("GA Runtime ", duration, " sec.")

# Plot Fitness and Number of Features versus Generation
plotGenerations(gen, lnfit, features)

# select the best individual
fit, individual, features = findBest(hof, goodFit, X, y)
print("{:-<40s}{:->39s}".format('*', '*'))
print("{:-<25s}   GA SELECTION BEST FEATURES   {:->22s}".format('*','*'))
print("{:-<25s} Best Fitness      {:.> 12.4f} {:->22s}".
                                      format('*', fit[0], '*'))
print("{:-<25s} Features Selected {:.> 12d} {:->22s}\n".
                                      format('*', len(features),'*'))

Xc = sm.add_constant(X[features])
model    = sm.Logit(y,Xc)  # Using StatsModels for Linear Regression
results  = model.fit()

ll       = model.loglike(results.params)
model_df = model.df_model + 2 #plus intercept and sigma
nobs     = y.shape[0]
aic      = em.aic(ll,  nobs, model_df)
bic      = em.bic(ll,  nobs, model_df)
aicc     = em.aicc(ll, nobs, model_df)

# These are the correct values as reported in SAS

print("\n{:=<40s}{:=>39s}".format('*', '*'))
print("AICC {:.> 19.3f}   AIC {:.> 20.3f}   BIC {:.>20.3f}\n".
      format(aicc, aic, bic))
print(results.summary())
lr = LogisticRegression(penalty='none').fit(X[features], y)
logreg.display_metrics(lr, X[features], y)
print("{:=<40s}{:=>39s}".format('*', '*'))

print("{:-<25s} GA SELECTION 70/30 VALIDATION {:->23s}".format('*','*'))

Xt, Xv, yt, yv = train_test_split(X[features], y, train_size=0.7, 
                                             random_state=12345)
lrg = LogisticRegression()
lrg = lr.fit(Xt, yt)
logreg.display_split_metrics(lrg, Xt, yt, Xv, yv)

#**************** STEPWISE SELECTION *****************************************
heading('Logistic Regression with Stepwise Selection')  
sw         = stepwise(encoded_df, target, reg="logistic", method="stepwise", 
                  crit_in=0.1, crit_out=0.1, verbose=True)
selected   = sw.fit_transform()
X_selected = encoded_df[selected]
Xc         = sm.add_constant(X_selected)
model      = sm.Logit(y,Xc)  # Using StatsModels for Linear Regression
results    = model.fit()

ll       = model.loglike(results.params)
model_df = model.df_model + 2 #plus intercept and sigma
nobs     = y.shape[0]
aic      = em.aic(ll,  nobs, model_df)
bic      = em.bic(ll,  nobs, model_df)
aicc     = em.aicc(ll, nobs, model_df)

# These are the correct values as reported in SAS

print("\n{:=<40s}{:=>39s}".format('*', '*'))
print("AICC {:.> 19.3f}   AIC {:.> 20.3f}   BIC {:.>20.3f}\n".
      format(aicc, aic, bic))
print(results.summary())
lr = LogisticRegression(penalty='none').fit(X_selected, y)
logreg.display_metrics(lr, X_selected, y)

heading('Stepwise 70/30 Validation')
Xt, Xv, yt, yv = train_test_split(X_selected, y, train_size=0.7, 
                                  random_state=12345)
lr = LogisticRegression(penalty='none')
lr = lr.fit(Xt, yt)
logreg.display_split_metrics(lr, Xt, yt, Xv, yv)

#**************** DECISION TREE      *****************************************  
heading('Decision Tree with Cross Validation')
depths = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
score_list = ['precision_macro', 'recall_macro', 'f1_macro']

# best_depth=17
best = 0

# dtc = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=5,   
#                              min_samples_split=5, random_state=12345)
# dtc.fit(X, y)
# tree_classifier.display_importance(dtc, X.columns,top=15, plot=True)
# tree_classifier.display_metrics(dtc, X, y)

for d in depths:
    print("\nTree Depth: ", d)
    dtc = DecisionTreeClassifier(max_depth=d, 
                                 min_samples_leaf=5, 
                                 min_samples_split=5,
                                 random_state=12345)
    dtc = dtc.fit(X,y)
    scores = cross_validate(dtc, X, y, scoring=score_list, 
                            return_train_score=False, cv=5)
    
    print("{:.<18s}{:>6s}{:>13s}".format("Metric", "Mean", 
                                          "Std. Dev."))
    for s in score_list:
        var = "test_"+s
        mean = scores[var].mean()
        std  = scores[var].std()
        print("{:.<18s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        if s=='f1_macro' and mean>best:
            best = mean
            best_depth_dt = d
            best_tree = deepcopy(dtc) # copies dtc and all nested objects


print("\nBest Tree Depth: ", best_depth_dt)
title = 'Best Decision Tree: Depth = ' + str(best_depth_dt)
tree_classifier.display_importance(best_tree, X.columns.values, 
                                top=15, plot=True)
tree_classifier.display_metrics(best_tree, X, y)


# print("\n{:-<25s} TREE 70/30 VALIDATION {:->27s}".format('*','*'))
# Xt, Xv, yt, yv = train_test_split(X_selected, y, train_size=0.7, 
#                                   random_state=12345)
# dtc = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=5,   
#                              min_samples_split=5, random_state=12345)
# dtc.fit(Xt, yt)
# tree_classifier.display_importance(dtc, Xt.columns, 
#                                    top=15, plot=True)
# tree_classifier.display_split_metrics(dtc, Xt, yt, Xv, yv)

#**************** RANDOM FOREST      *****************************************  
# # Best Tree with Depth=17
# best_depth = 17
# rfc = RandomForestClassifier(n_estimators=100, max_depth=best_depth, 
#                              min_samples_leaf=5, min_samples_split=5,  
#                              random_state=12345)
# rfc.fit(X, y)
# forest_classifier.display_importance(rfc, X.columns,top=15, plot=True)
# forest_classifier.display_metrics(rfc, X, y)

# print("\n{:-<25s} RANDOM FOREST 70/30 VALIDATION {:->27s}".format('*','*'))
# Xt, Xv, yt, yv = train_test_split(X_selected, y, train_size=0.7, 
#                                   random_state=12345)
# rfc = RandomForestClassifier(n_estimators=100, max_depth=best_depth, 
#                              min_samples_leaf=5, min_samples_split=5,  
#                              random_state=12345)
# rfc.fit(Xt, yt)
# forest_classifier.display_importance(rfc, Xt.columns, 
#                                    top=15, plot=True)
# forest_classifier.display_split_metrics(rfc, Xt, yt, Xv, yv)

heading('RANDOM FOREST K-FOLD VALIDATION')
best = 0


for d in depths:
    print("\nTree Depth: ", d)
    rfc = RandomForestClassifier(n_estimators=100, max_depth=d, 
                             min_samples_leaf=5, min_samples_split=5,  
                             random_state=12345)

    rfc = rfc.fit(X,y)
    scores = cross_validate(rfc, X, y, scoring=score_list, 
                            return_train_score=False, cv=5)
    
    print("{:.<18s}{:>6s}{:>13s}".format("Metric", "Mean", 
                                          "Std. Dev."))
    for s in score_list:
        var = "test_"+s
        mean = scores[var].mean()
        std  = scores[var].std()
        print("{:.<18s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        if s=='f1_macro' and mean>best:
            best = mean
            best_depth_rf = d
            best_forest = deepcopy(rfc) # copies dtc and all nested objects


print("\nBest Forest Depth: ", best_depth_rf)
title = 'Best Random Forest: Depth = ' + str(best_depth_rf)
forest_classifier.display_importance(best_forest, X.columns.values, 
                                top=15, plot=True)
forest_classifier.display_metrics(best_forest, X, y)



#**************** ENSEMBLE           ***************************************** 
estimators=[
    ('Random Forest', rfc),
    ('Decision Tree', dtc),
    ('Log. Reg', lrg)
    ]
ecl = VotingClassifier(estimators=estimators, voting='soft' )
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    eclf = ecl.fit(Xt,yt)
    for clf, label in zip([lrg, lr, dtc, lrg, eclf], ['GA Logistic Regression',
                                                      'SW Logistic Regression',
                                                      'Decision Tree', 
                                                      'Random Forest', 
                                                      'Ensemble']):
        scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), \
                                                    scores.std(), label))   
    # probas = [c.fit(X, y).predict_proba(X) for c in estimators]


############################################################################
## REMOVE 'TITLE' FROM tree_classifier.display_importance AND             ##
## FROM forest_classifier.display_importance AS IT IS NOT COMPATIBLE WITH ##
## STANDARD IMPLEMENTATIONS OF AdvancedAnalytics                          ##
############################################################################