# Kimberly Kaminsky - Assignment #3
# Boston Housing Study (Python)

# Here we use data from the Boston Housing Study to evaluate
# regression modeling methods within a cross-validation design.

#################### 
# Define constants #
####################

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 111

# ten-fold cross-validation employed here
N_FOLDS = 20

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True

####################
# Import Libraries #
####################

# import base packages into the namespace for this program
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt           # static plotting
import seaborn as sns                     # pretty plotting, including heat map
import matplotlib as mpl
import PyPDF2 as pp                      # Allows pdf file manipulation

# modeling routines from Scikit Learn packages
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score  
from math import sqrt, log  # for root mean-squared error calculation

                                                  
# specify the k-fold cross-validation design
from sklearn.model_selection import KFold

################################
# Functions for use in program #
################################

# function to clear output console
def clear():
    print("\033[H\033[J")
    
# correlation heat map setup for seaborn
def corr_chart(df_corr):
    corr=df_corr.corr()
    #screen top half to get a triangle
    top = np.zeros_like(corr, dtype=np.bool)
    top[np.triu_indices_from(top)] = True
    fig=plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, mask=top, cmap='coolwarm', 
        center = 0, square=True, 
        linewidths=.5, cbar_kws={'shrink':.5}, 
        annot = True, annot_kws={'size': 9}, fmt = '.3f')           
    plt.xticks(rotation=45) # rotate variable labels on columns (x axis)
    plt.yticks(rotation=0) # use horizontal variable labels on rows (y axis)
    plt.title('Correlation Heat Map')   
    plt.savefig(datapath+'corrmap.pdf') 

def scatterplot(column1, column2):   
    fig = plt.figure()
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title("Scatterplot: '%s' vs '%s'" % (column1, column2))
    plt.scatter(boston[column1], 
        boston[column2],
        facecolors = 'none', 
        edgecolors = 'blue') 
    pdf.savefig(fig)
    
def histogram(column):
    fig = plt.figure()
    plt.hist(boston[column].dropna(), facecolor='blue')
    plt.title(column)
    plt.xlabel(column)
    plt.grid()
    plt.axis([0, max(boston[column]), 0, len(boston)])
    pdf.savefig(fig)    

def boxplt(column1):    
    fig = plt.figure()
    plt.xlabel(column1)
    plt.title("Boxplot: %s" % column1)
    plt.boxplot(boston[column1].reset_index(drop = True)) 
    pdf.savefig(fig) 

#########################
# Import and clean data #
#########################

# Import the dataset
datapath = os.path.join("D:/","Kim MSPA", "Predict 422", "Assignments", "Assignment3", "")
boston_input = pd.read_csv(datapath + "boston.csv")

# check the pandas DataFrame object boston_input
print('\nboston DataFrame (first and last five rows):')
print(boston_input.head())
print(boston_input.tail())

print('\nGeneral description of the boston_input DataFrame:')
print(boston_input.info())

# drop neighborhood from the data being considered
boston = boston_input.drop('neighborhood', 1)
print('\nGeneral description of the boston DataFrame:')
print(boston.info())

print('\nDescriptive statistics of the boston DataFrame:')
print(boston.describe())

# Create an array containing the log valus for the response variable
mvNumpy = np.array(boston['mv'])
mvLog = [log(y) for y in mvNumpy]

# set up preliminary data for data for fitting the models 
# the first column is the median housing value response
# the remaining columns are the explanatory variables
prelim_model_data = np.array([mvLog,\
    boston.crim,\
    boston.zn,\
    boston.indus,\
    boston.chas,\
    boston.nox,\
    boston.rooms,\
    boston.age,\
    boston.dis,\
    boston.rad,\
    boston.tax,\
    boston.ptratio,\
    boston.lstat]).T

# dimensions of the polynomial model X input and y response
# preliminary data before standardization
print('\nData dimensions:', prelim_model_data.shape)

# standard scores for the columns... along axis 0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(prelim_model_data))
# show standardization constants being employed
print(scaler.mean_)
print(scaler.scale_)

# the model data will be standardized form of preliminary model data
model_data = scaler.fit_transform(prelim_model_data)

# dimensions of the polynomial model X input and y response
# all in standardized units of measure
print('\nDimensions for model_data:', model_data.shape)

######################
# Data Visualization #
######################

# Turn off interactive mode since a large number of plots are generated
# Plots will be saved off in pdf files
mpl.is_interactive()
plt.ioff()

# examine intercorrelations among software preference variables
# with correlation matrix/heat map
corr_chart(df_corr = boston) 

# Create list of column names for boston data set
boston_col_names = boston.columns

# Create master pdf for saving the figures
pdf = mpl.backends.backend_pdf.PdfPages(datapath + "output.pdf")

# Scatterplots comparing course interest with software preference
for x in range(0, len(boston_col_names)):
    for y in range(0, len(boston_col_names)):
        if (x != y):
            scatterplot(boston_col_names[x], boston_col_names[y])

# box plots to check for outliers
for x in range(0, len(boston_col_names)):
    boxplt(boston_col_names[x])
 
# check distributions of the variables using histogram
for x in range(0, len(boston_col_names)):
    histogram(boston_col_names[x]) 

# histogram of the log of response variable
fig = plt.figure()
plt.hist(mvLog, facecolor='blue')
plt.title('Log of Housing Values')
plt.xlabel("mvLog")
plt.grid()
plt.axis([0, max(mvLog), 0, len(mvLog)])
pdf.savefig(fig) 
     
pdf.close()


#################################
# Select best modeling method  #
#################################


# specify the set of regressors being evaluated
names = ['Linear_Regression', 'Ridge_Alpha1', 'Ridge_Alpha10', 
         'Ridge_Alpha_20', 'Lasso_Alpha_point1', 'Lasso_Alpha_point01',
         'Lasso_Alpha_point001', 'ElasticNet_Alpha_point1', 
         'ElasticNet_Alpha_point01', 'ElasticNet_Alpha_point001'] 
regressors = [LinearRegression(fit_intercept = SET_FIT_INTERCEPT),
              Ridge(alpha = 1, solver = 'cholesky', 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     normalize = False, 
                     random_state = RANDOM_SEED), 
               Ridge(alpha = 10, solver = 'cholesky', 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     normalize = False, 
                     random_state = RANDOM_SEED),
              Ridge(alpha = 20, solver = 'cholesky', 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     normalize = False, 
                     random_state = RANDOM_SEED),
               Lasso(alpha = 0.1, max_iter=10000, tol=0.01, 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     random_state = RANDOM_SEED),
               Lasso(alpha = .01, max_iter=10000, tol=0.01, 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     random_state = RANDOM_SEED),
               Lasso(alpha = .001, max_iter=10000, tol=0.01, 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     random_state = RANDOM_SEED),                                          
               ElasticNet(alpha = 0.1, l1_ratio = 0.5, 
                          max_iter=10000, tol=0.01, 
                          fit_intercept = SET_FIT_INTERCEPT, 
                          normalize = False, 
                          random_state = RANDOM_SEED),
               ElasticNet(alpha = .01, l1_ratio = 0.5, 
                          max_iter=10000, tol=0.01, 
                          fit_intercept = SET_FIT_INTERCEPT, 
                          normalize = False, 
                          random_state = RANDOM_SEED),
               ElasticNet(alpha = .001, l1_ratio = 0.5, 
                          max_iter=10000, tol=0.01, 
                          fit_intercept = SET_FIT_INTERCEPT, 
                          normalize = False, 
                          random_state = RANDOM_SEED)]

# set up numpy array for storing results
cv_results = np.zeros((N_FOLDS, len(names)))

kf = KFold(n_splits = N_FOLDS, shuffle=False, random_state = RANDOM_SEED)

# check the splitting process by looking at fold observation counts
index_for_fold = 0  # fold count initialized 

for train_index, test_index in kf.split(model_data):
    print('\nFold index:', index_for_fold,
          '------------------------------------------')
#   the structure of modeling data for this study has the
#   response variable coming first and explanatory variables later          
#   so 1:model_data.shape[1] slices for explanatory variables
#   and 0 is the index for the response variable    
    X_train = model_data[train_index, 1:model_data.shape[1]]
    X_test = model_data[test_index, 1:model_data.shape[1]]
    y_train = model_data[train_index, 0]
    y_test = model_data[test_index, 0]   
    print('\nShape of input data for this fold:',
          '\nData Set: (Observations, Variables)')
    print('X_train:', X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:',y_test.shape)

    index_for_method = 0  # initialize
    for name, reg_model in zip(names, regressors):
        print('\nRegression model evaluation for:', name)
        print('  Scikit Learn method:', reg_model)
        reg_model.fit(X_train, y_train)  # fit on the train set for this fold
        print('Fitted regression intercept:', reg_model.intercept_)
        print('Fitted regression coefficients:', reg_model.coef_)
 
        # evaluate on the test set for this fold
        y_test_predict = reg_model.predict(X_test)
        print('Coefficient of determination (R-squared):',
              r2_score(y_test, y_test_predict))
        fold_method_result = sqrt(mean_squared_error(y_test, y_test_predict))
        print(reg_model.get_params(deep=True))
        print('Root mean-squared error:', fold_method_result)
        cv_results[index_for_fold, index_for_method] = fold_method_result
        index_for_method += 1
  
    index_for_fold += 1

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.columns = names

print('\n----------------------------------------------')
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      'in standardized units (mean 0, standard deviation 1)\n',
      '\nMethod               Root mean-squared error', sep = '',)     
print(cv_results_df.mean())   


# Now run model on full dataset
ridge20 = Ridge(alpha = 20, solver = 'cholesky', 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     normalize = False, 
                     random_state = RANDOM_SEED)

ridge20.fit(model_data[:, 1:13], model_data[:, 0])  
print('Fitted regression intercept:', ridge20.intercept_)
print('Fitted regression coefficients:', ridge20.coef_)

predictions = ridge20.predict(model_data[:, 1:13])
rmse = sqrt(mean_squared_error(model_data[:, 0], predictions))
print("ridge20 RMSE: ", rmse)


#########################################
# Combine all pdf files into 1 pdf file #
#########################################

pdfWriter = pp.PdfFileWriter()
pdfOne = pp.PdfFileReader(open(datapath + "corrmap.pdf", "rb"))
pdfTwo = pp.PdfFileReader(open(datapath + "output.pdf", "rb"))

for pageNum in range(pdfOne.numPages):        
    pageObj = pdfOne.getPage(pageNum)
    pdfWriter.addPage(pageObj)


for pageNum in range(pdfTwo.numPages):        
    pageObj = pdfTwo.getPage(pageNum)
    pdfWriter.addPage(pageObj)


outputStream = open(datapath + r"Assign3_Output.pdf", "wb")
pdfWriter.write(outputStream)
outputStream.close()






