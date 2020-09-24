"""
Created 03 JUN 2020

@author: el-rainwater
"""

import pandas as pd
import numpy as np
from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT
from AdvancedAnalytics.Regression import linreg, stepwise
import statsmodels.api as sm
import statsmodels.tools.eval_measures as em
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


file = 'OilProductionHW2.xlsx'
filepath = 'C:/Users/rainwater-e/OneDrive - Texas A&M University/' \
    'Summer-2020/STAT 656 Applied Analytics/hw-02/'

df = pd.read_excel(filepath + file)
print(df.head(), df.shape)
print(df.dtypes)

attribute_map = {
    'Obs':[DT.ID, (1, np.inf)],
    'Log_Cum_Production':[DT.Interval, (8,15)],
    'Log_Proppant_LB':[DT.Interval, (6, 18)],
    'Log_Carbonate':[DT.Interval, (-4, 4)],
    'Log_Frac_Fluid_GL':[DT.Interval, (7, 18)],
    'Log_GrossPerforatedInterval':[DT.Interval, (4,9)],
    'Log_LowerPerforation_xy':[DT.Interval, (8,10)],
    'Log_UpperPerforation_xy':[DT.Interval, (8,10)],
    'Log_TotalDepth':[DT.Interval, (8,10)],
    'N_Stages':[DT.Interval, (2,14)],
    'X_Well':[DT.Interval, (-100, -95)],
    'Y_Well':[DT.Interval, (30,35)],
    'Operator':[DT.Nominal, list(range(1,29))],
    'County':[DT.Nominal, list(range(1,15))]
}

target = 'Log_Cum_Production'

# One-hot encode and impute missing values
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot',
                          no_impute=[target], interval_scale=None, drop=True,
                          display=True)
df_encoded = rie.fit_transform(df).dropna() #drop rows with missing values


# Set up stepwise regression
sw = stepwise(df_encoded, target, reg='linear', method='stepwise',
              crit_in=0.1, crit_out=0.1, verbose=True)

selected = sw.fit_transform()

print('\nFinal selected attributes:\n', selected)

y = df_encoded[target]
y = np.ravel(y) # Ravel it into a contiguous flattened array

X  = df_encoded[selected] # create a dataframe of the selected attributes
Xc = sm.add_constant(X)   # Add an intercept column to the regressors so that
                          # sm.OLS will work correctly
model   = sm.OLS(y,Xc)    # Use StatsModels.OLS
results = model.fit()

ll       = model.loglike(results.params) # Returns the log likelihood function
model_df = model.df_model + 2            # Corrects the DOF by adding for the
                                         # intercept and sigma
nobs     = y.shape[0]                    # Returns number of observations
aic      = em.aic(ll, nobs, model_df)
bic      = em.bic(ll, nobs, model_df)
aicc     = em.aicc(ll, nobs, model_df)

predicted  = results.fittedvalues
residual   = results.resid
influence  = results.get_influence()


# Gonna just copy this formatty stuff from Dr. J's example
# These are the correct values as reported in SAS
print("\n")
print("AICC: ", aicc, "AIC: ", aic, "BIC: ", bic)

print("\n***********************************************************")
print("                                   Target: " + target)
print(results.summary())
print("\n***********************************************************")


# Determine the attribute with the greatest absolute test statistic value
max_tvalue = results.tvalues[results.tvalues.keys()!='const'].abs().max()
max_attrib = results.tvalues[results.tvalues==max_tvalue].index[0]
print("\nStrongest Attribute: " + max_attrib)

# Set sigma intervals
n3 = 2.0*(1.0-norm.cdf(3.0)) * nobs
n4 = 2.0*(1.0-norm.cdf(4.0)) * nobs
n5 = 2.0*(1.0-norm.cdf(5.0)) * nobs
n6 = 2.0*(1.0-norm.cdf(6.0)) * nobs

print("\nExpected number of observations outside stated limits:")
print("-+ 3Sigma: ", int(round(n3)))
print("-+ 4Sigma: ", int(round(n4)))
print("-+ 5Sigma: ", int(round(n5)))

print("\n")
leverage  = influence.hat_matrix_diag
cooks_d   = influence.cooks_distance[0]
        
cutoffD = 4.0/nobs
print("Max Cooks D: {:8.4f}  Cutoff ( 4/n): {:8.5f}".
      format(cooks_d.max(), cutoffD))
cutoffH = 2.0 * model_df / nobs
print("Max H:       {:8.4f}  Cutoff (2p/n): {:8.5f}".
      format(leverage.max(), cutoffH))

std_residuals  = residual/np.sqrt(results.mse_resid)
stud_residuals = influence.resid_studentized_internal

print("\nResiduals beyond 4 sigma:")
outliers = np.nonzero(stud_residuals > 4)[0]
outliers = np.append(outliers, np.nonzero(stud_residuals<-4)[0])
print("Total Number of Outliers:", outliers.shape[0])
print("\nFirst Fifteen Residuals beyond 4 sigma:")
print("\n*******************************************************")
print(" Case      Observed      Predicted       Stud. Resid.")
cases = 0
for case in outliers:
        print("{:5d}{:13.2f}{:14.2f}{:17.2f}". 
              format(case, y[case], predicted[case], stud_residuals[case]))
        cases += 1
        if cases==15: break
print("")

print("\n*******************************************************")
print("                             Min      Mean        Max")
print("Observed:              {:10.4f} {:10.4f} {:10.4f}".
      format(y.min(),  y.mean(), y.max()))
print("Predicted:             {:10.4f} {:10.4f} {:10.4f}".
      format(predicted.min(),  predicted.mean(), predicted.max()))
print("Residuals:            {:10.4f} {:10.4f} {:10.4f}".
      format(residual.min(),  residual.mean(), residual.max()))
print("Standardized Residuals:{:10.4f} {:10.4f} {:10.4f}".
      format(std_residuals.min(),std_residuals.mean(),std_residuals.max()))
print("Studentized  Residuals:{:10.4f} {:10.4f} {:10.4f}".
      format(stud_residuals.min(),stud_residuals.mean(),stud_residuals.max()))
print("Cooks'D:               {:10.4f} {:10.4f} {:10.4f}".
      format(cooks_d.min(),  cooks_d.mean(), cooks_d.max()))
print("*******************************************************")
# Using MatPlot for residual and influence graphics
# Plot of Observed Values versus the Predicted Values
plt.figure()
plt.xlabel("Predicted " + target)
plt.ylabel("Observed " + target)
plt.plot(predicted, y, "bo")
plt.plot( [8,16], [8,16], "r", linewidth=1, markersize=9)
plt.axis([8, 16, 8, 16])
plt.show()

# Multiplot of 1.  Obs Number vs Studentized Residuals, 
#              2.  Predicted Value vs. Studentized Residuals, and
#              3.  Cook's D vs. Studentized Residuals
plt.figure()
plt.subplots(figsize=(16,12))
plt.subplot(311)
plt.xlabel(max_attrib)
plt.ylabel("Studentized Residual")
plt.axis([8,10, -10.0, +10.0])
plt.axhline(0, color="k")
plt.axhline(4, color="r")
plt.axhline(-4, color="r")
plt.plot(df_encoded[max_attrib], stud_residuals, "bo")

plt.subplot(312)
plt.xlabel("Predicted " + target)
plt.ylabel("Studentized Residual")
plt.axis([8, 16, -10.0, +10.0])
#plt.axis([4, 14, -10.0, +10.0])
plt.axhline(0, color="k")
plt.axhline(4, color="r")
plt.axhline(-4, color="r")
plt.plot(predicted, stud_residuals, 'bo')

plt.subplot(313)
plt.xlabel("Cooks'D")
plt.ylabel("Studentized Residual")
plt.axis([0,0.03, -10.0, +10.0])
plt.axhline(0, color="k")
plt.axhline(4, color="r")
plt.axhline(-4, color="r")
plt.axvline(cutoffD, color="g")
plt.plot(cooks_d, stud_residuals, 'bo')
cutoffText = 'Cutoff = (4/n) = {0:.4f}'.format(cutoffD)
plt.text(cutoffD+0.00005, -9, cutoffText, fontsize=12)
plt.show()

# Split the model 70/30 for training/validation

X_train, X_validate, y_train, y_validate =  \
    train_test_split(Xc, y, test_size=0.3, random_state = 12345)

model    = sm.OLS(y_train, X_train)  # Using StatsModels for Linear Regression
results  = model.fit()
print("\n\n*******************************************************")
print("                              TRAINING MODEL")
print("                              Target: " + target)
print(results.summary())
print("\n*******************************************************\n")

print("AdvancedAnalytics Display Split Metrics:")
linreg.display_split_metrics(results, X_train,y_train, X_validate,y_validate)


print("\n*******************************************************")
print("                             Min      Mean        Max")
print("Observed:              {:10.4f} {:10.4f} {:10.4f}".\
      format(y.min(),  y.mean(), y.max()))
print("Predicted:             {:10.4f} {:10.4f} {:10.4f}".\
      format(predicted.min(),  predicted.mean(), predicted.max()))
print("Residuals:             {:10.4f} {:10.4f} {:10.4f}".\
      format(residual.min(),  residual.mean(), residual.max()))
print("Studentized Residuals: {:10.4f} {:10.4f} {:10.4f}".\
      format(stud_residuals.min(),stud_residuals.mean(),stud_residuals.max()))
print("Cooks'D:               {:10.4f} {:10.4f} {:10.4f}".\
      format(cooks_d.min(),  cooks_d.mean(), cooks_d.max()))
print("*******************************************************")

diag_df = pd.DataFrame()
diag_df['row']       = range(nobs)  
diag_df['observed']  = y 
diag_df['predicted'] = predicted
diag_df['residuals'] = residual
diag_df['leverage']  = leverage
diag_df['cooks_d']   = cooks_d
diag_df['stud_residuals'] = stud_residuals
diag_df.sort_values(by=["observed"], ascending=False, inplace=True)

print("\n          Top 10 " + target)
print(" Obs.   Observed    Predicted")
for i in range(10):
    print("{:5d}{:9.2f}{:13.2f}".format(diag_df["row"].iloc[i], \
          diag_df["observed"].iloc[i], \
          diag_df["predicted"].iloc[i]))