from scipy import stats
stats.uniform(0, 1).rvs()
stats.bernoulli(0.6).rvs()

#plots
------------------------------------------
import matploblib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

from pandas.plotting import scatter_matrix
#diagonal='kde' changes 斜向的 histogram 改成了线 图
_ = scatter_matrix(concrete, alpha=0.2, figsize=(20, 20), diagonal='kde')

#changes pandas series to numpy array
y = grad['admit'].values
variable_names = ['gre', 'gpa', 'rank']

# generate jitter, 数量和 df 数据行数一样 len(grad)
jitter = stats.uniform(-0.03,0.06).rvs(len(grad))
fig, axs = plt.subplots(1, 3, figsize = (14, 4))

#variable_names 一定要有3 个，因为 axs 有 1行，3 个
for variable, ax in zip(variable_names, axs.flatten()):
    ax.scatter(grad[variable], y + jitter, s=10, alpha=0.1)
    ax.text(0.1, 0.4, 'Ride the light!', fontsize = 35)
fig.tight_layout()
------------------------------------------
#provide random test data

#Random values in a given shape.

#Create an array of the given shape and populate it with random samples from 
# a uniform distribution over [0, 1).
np.random.rand(3,2)


#20% of them are 1, 80% of the random points are 0
jitter = stats.uniform(-0.03,0.06).rvs(len(grad))

y = stats.bernoulli(0.2).rvs(npts)
#出来的y 是一大串'Not', 和 'fraud'
y = np.array(['Not', 'Fraud'])[y]
#plot different points with different colors
y = np.array(['red', 'yellow'])[y]

#等距离的几个数字
>>> np.linspace(2.0, 3.0, num=5)
array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])

------------------------------------------
#models

#model_selection is a py file, train_test_split is a function in the .py file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
model = sklearn.linear_model.LogisticRegression()
#model = sklearn.neighbors.KNeighborsClassifier(50)
model.fit(Xtrain, ytrain)

(model.predict(Xtest) == ytest).mean()
#this is accuracy for 0, 1 classification


----------------------------------------------


'''ROC curve'''
import numpy as np
def roc_curve(probabilities, y_test):
    '''
        Sort instances by their prediction strength (the probabilities)
    For every instance in increasing order of probability:
        Set the threshold to be the probability
        Set everything above the threshold to the positive class
        Calculate the True Positive Rate (aka sensitivity or recall)
        Calculate the False Positive Rate (1 - specificity)
    Return three lists: TPRs, FPRs, thresholds
    '''
    tpr, fpr, thresholds = [], [], []
    instances = [(probability, y) for probability, y in zip(probabilities, y_test)]
    instances.sort(key = lambda t: t[0])
    total_zeros = len([x[1] for x in instances if x[1] == 0])
    total_ones = len([x[1] for x in instances if x[1] == 1])
    thresholds = [instance[0] for instance in instances]
    for i in range(len(instances)):
        tp = np.sum([x[1] for x in instances[i:]])
        temp_tpr = tp / total_ones
        temp_fpr = (len(instances[i:]) - tp) / total_zeros
        tpr.append(temp_tpr)
        fpr.append(temp_fpr)
    
    return tpr, fpr, thresholds
roc_curve([0.1, 0.3, 0.9, 0.2], [1, 0, 0, 1])

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Generate a random n-class classification problem.
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=2, n_samples=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]

tpr, fpr, thresholds = roc_curve(probabilities, y_test)
plt.figure(figsize=(15,5))
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity, Recall)")
plt.title("ROC plot of fake data")
for i in range(0, len(fpr), 10):
    plt.text(fpr[i], tpr[i] - 0.01, np.round(thresholds[i],2), fontsize=10)
plt.show()

'''ROC curve'''


 pd.DataFrame(data=data[1:,1:],    # values
...              index=data[1:,0],    # 1st column as index
...              columns=data[0,1:])  # 1st row as the column names 


#check NAs

#convert sklearn dataset to pandas dataframe
df = pd.DataFrame(data = diabetes_attr[:100, :], columns = diabetes.feature_names)
na_df = df.isna()
na_df.sum(axis = 0)

df = df.dropna(axis = 1)
df.describe()

#fill NA with 0
df2.fillna(0)

#fill NAs in one column
df2['one'].fillna('missing')


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data # housing features
y = boston.target # housing prices
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
def rmse(true, predicted):
    return np.sqrt(sum((predicted - true)**2)/len(true))
#Train-Test Validation (BAD OPTION):
# # Fit your model using the training set
knr = KNeighborsRegressor()
# knr.fit(X_train, y_train)
# # Call predict to get the predicted values for training and test set
# train_predicted = knr.predict(X_train)
# test_predicted = knr.predict(X_test)
# # Calculate RMSE for training and test set
# rmse(train_predicted, y_train)
# rmse(test_predicted, y_test)

#K-fold Cross Validation (BETTER OPTION):
def crossVal(X, y, k=5):
    kf = KFold(n_splits=k, shuffle = True)
    error = []
    #train_index is the a list of row numbers that are selected for X_train
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knr.fit(X_train, y_train)
        #train_predicted = knr.predict(X_train)
        test_predicted = knr.predict(X_test)
        error.append(rmse(y_test, test_predicted))
    return np.mean(error)
print(crossVal(X, y, 100))






#make confusion matrix
#return 每个种类的个数
from sklearn.metrics import confusion_matrix
def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])


##Regularization

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def cv(X, y, base_estimator, n_folds, kwargs, random_seed=154):
    """Estimate the in- and out-of-sample error of a model using cross
    validation.
    
    Parameters
    ----------
    
    X: np.array
      Matrix of predictors.
      
    y: np.array
      Target array.
      
    base_estimator: sklearn model object.
      The estimator to fit.  Must have fit and predict methods.
      
    n_folds: int
      The number of folds in the cross validation.
      
    random_seed: int
      A seed for the random number generator, for repeatability.
    
    Returns
    -------
      
    train_cv_errors, test_cv_errors: tuple of arrays
      The training and testing errors for each fold of cross validation.
    """
    
    kf = KFold(n_splits=n_folds, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    #MSEs = []
    for idx, (train, test) in enumerate(kf.split(X)):
        # Split into train and test
        #X_train = X.iloc[train, :]
        #train and test are lists of row numbers being selected
        X_train = X[train]
        y_train = y[train]
        y_test = y[test]
        X_test = X[test]
        # Standardize data, fit on training set, transform training and test data, Don't fit test data
        model = Pipeline([('standardize', StandardScaler()),
                   ('regressor', base_estimator(**kwargs))])
        
        # Fit ridge regression to training data.
        
        model.fit(X_train, y_train)

        # Make predictions.
        #this will transform the test data automatically, pipeLine yay!
        predictions = model.predict(X_test)
        # Calculate MSE.
        MSEy = np.mean((y_test - predictions) ** 2.0)
        MSEx = np.mean((y_train - model.predict(X_train))**2.0)
        #MSEs.append(np.mean(MSE))
        train_cv_errors[idx] = MSEx
        test_cv_errors[idx] = MSEy
    return train_cv_errors, test_cv_errors
kwargs = {"alpha":0.5, "fit_intercept":True}
cv(diabetes.data, diabetes.target, Lasso, 10, kwargs)



##check if you model is worse than just predicting mean
y_dummy = np.ones(len(y_test)) * mean_price
er_dummy = np.sqrt(mean_squared_log_error(y_test,y_dummy))

#engineer new features based on one feature
size_fit = Pipeline([
    ('size', ColumnSelector(name='ProductSize')),
  
    ('size_features', FeatureUnion([
        # MapFeature allows us to use a function on a column to create a new feature.
        ('size_nan', MapFeature(lambda size: pd.isnull(size), 'size nan')),
        ('size_small', MapFeature(lambda size: (size=='Small')|(size=='Compact'), 'size_small')), 
        ('size_mini', MapFeature(lambda size: size=='Mini', 'size_mini')),
        ('size_med', MapFeature(lambda size: (size=='Medium')|(pd.isnull(size)), 'size_med')), 
        ('size_med_large', MapFeature(lambda size: size=='Large / Medium', 'size_med_large')),
        ('size_large', MapFeature(lambda size: size=='Large', 'size_large'))
        
    ]))
])

size_fit.transform(data).head()

feature_pipeline = FeatureUnion([
    ('intercept', Intercept()),
    ('size_fit', size_fit)
])

model_siz = Pipeline([('cleaning', feature_pipeline),
                   ('regressor', LinearRegression())])

----------------------------------------------------
#Other things

#cross-tab count
pd.crosstab(ytest, model.predict(Xtest), rownames=['actual'], colnames=['predicted'])

#replace spaced in column names with '_'
df.columns = [col.lower().replace(' ', '_') for col in df.columns]


----------------------------------------------------
#Calculate error
from sklearn.metrics import mean_squared_log_error, mean_squared_error

er_tr = np.sqrt(mean_squared_log_error(y_train,y_pred_tr))
er_test = np.sqrt(mean_squared_log_error(y_test,y_pred))
er_tr,er_test

----------------------------------------------------
#how to standardize data

def standardize_y(y_train, y_test):
    y_mean, y_std = np.mean(y_train), np.std(y_train)
    y_train_std = (y_train - y_mean) / y_std
    y_test_std = (y_test - y_mean) / y_std
    return y_train_std, y_test_std



y_train, y_test = standardize_y(train_raw["Balance"], test_raw["Balance"])

----------------------------------------------------
#how to pass ax into a function that draws a graph
def one_dim_scatterplot(data, ax, jitter=0.2, **options):
    if jitter:
        jitter = np.random.uniform(-jitter, jitter, size=data.shape)
    else:
        jitter = np.repeat(0.0, len(data))
    ax.scatter(data, jitter, **options)
    ax.yaxis.set_ticklabels([])
    ax.set_ylim([-1, 1])


#Jacob's graph example, case study1

def convert_to_numeric(catagorical):
    classes = catagorical.unique()
    classes_mapping = {cls: i for i, cls in enumerate(classes)}
    classes_inv_mapping = {i: cls for i, cls in enumerate(classes)}
    classes_numeric = catagorical.apply(lambda cls: classes_mapping[cls])
    return classes_numeric, classes_inv_mapping

    classes = categories.unique()
    class_map = {clss:i for i, cls in enumerate(classes)}
    class_inv = {i: clss for i, clss in enumerate(classes)}
    class_dummies = categories.apply(lambda cls: class_map[cls])
    return class_dummies


fig, ax = plt.subplots(figsize=(50, 50))

catagorical = df_truncated['ProductGroupDesc']
y = df_truncated['SalePrice']

numeric, classes_mapping = convert_to_numeric(catagorical)

noise = np.random.uniform(-0.3, 0.3, size=len(catagorical))
ax.scatter(numeric + noise, y, color="orange", alpha=0.05)

box_data = list(y.groupby(catagorical))
ax.boxplot([data for _, data in box_data], positions=range(len(box_data)))
ax.set_xticks(list(classes_mapping))
ax.set_xticklabels(list(catagorical.unique()))

ax.set_xlabel("Extras")
ax.set_ylabel("Sale Price")
ax.set_title("Univariate Extras/Price Graph")

mean_enclosure = [ box_data[i][1].mean() for i in range(6) ]
ax.plot(range(len(box_data)), mean_enclosure, color='b')
plt.savefig("Extras_v_Price.png")

#give you a new column with age in a particular bin / interval
pd.cut(df_signed_in['Age'], [7, 18, 24, 34, 44, 54, 64, 1000])

----------------------------------------------------------
#Fishers' Tea Problem
#build a small Bernoulli Distribution graph, make one bar color 'red'
bernoulli = spl.Binomial(6, 0.5)

fig, ax = plt.subplots(1, figsize=(10, 4))
bars = ax.bar(range(7), [bernoulli.pdf(i) for i in range(7)], align="center", color="grey")
bars[5].set_color('red')

binomial = spl.Binomial(6, 0.5)

# CDF give P(draw <= value)
prob_equal_or_more_extreme = 1 - binomial.cdf(4)
print("Probability of Obsrving Data More Equal or More Extreme than Actual: {:2.2}".format(
    prob_equal_or_more_extreme))

#make a normal line and fill the p-value region
fig, ax = plt.subplots(1, figsize=(16, 3))

ax.plot(x, normal_approx.pdf(x), linewidth=3)
ax.set_xlim(2400, 2600)
ax.fill_between(x, normal_approx.pdf(x), 
                where=(x >= 2530), color="red", alpha=0.5)
ax.set_title("p-value Reigon")


------------------------------------------------------

#null pypothesis graph

# null hypothesis - Bottles weights average =  20.4 ounces
#alternative hypothesis - Bottles weights average !=  20.4 ounces
# two tailed test
#Sample average should be population ave = 20.4, CLT
#we don't know the SD of the population, so we calculate the sd of the sample to approximate

fig, ax = plt.subplots(1, figsize = (15, 8))
x = np.linspace(20, 21, len(data))

SD = np.std(data)
sample_size = 130
sample_mean = 20.4
effect_size = 0.1
alpha = 0.05


h0 = stats.norm(sample_mean, SD / np.sqrt(sample_size))
ha = stats.norm(sample_mean + effect_size, SD / np.sqrt(sample_size))

#For this we use the quantile (or percent probability) function, which is the inverse of the cdf.
critical_value1 = stats.norm(sample_mean, SD / np.sqrt(sample_size)).ppf(alpha)
critical_value2 = stats.norm(sample_mean, SD / np.sqrt(sample_size)).ppf(1 - alpha)

xpos1 = x[x <= critical_value1]
xpos2 = x[x >= critical_value2]

xneg1 = x[x <= critical_value1]
xneg2 = x[x <= critical_value2]

ax.plot(x, h0.pdf(x), color = 'r', label = '$H_0$')
ax.plot(x, ha.pdf(x), color = 'b', label = '$H_a$')

ax.fill_between(xpos1, 0, h0.pdf(xpos1), color = 'r', alpha = 0.2, label = "$\\alpha$")
ax.fill_between(xpos2, 0, h0.pdf(xpos2), color = 'r', alpha = 0.2)

ax.fill_between(xneg1, 0, ha.pdf(xneg1), color = 'b', alpha = 0.2, label = "$\\beta$")
ax.fill_between(xneg2, 0, ha.pdf(xneg2), color = 'b', alpha = 0.2)

ax.fill_between(xpos2, 0, ha.pdf(xpos2), color = 'black', alpha = 0.2, hatch = '////', label = "power")
ax.axvline(critical_value1, color = 'black', label = 'critical value 1')
ax.axvline(critical_value2, color = 'brown', label = 'critical value 2')

ax.set_xlabel('sample mean')
ax.set_ylabel('pdf')
ax.set_ylim(ymin = 0.0)
ax.legend()

plt.show()

-------------------------------------------
# instead of using defautdict, use get method for python dictionaries
dd.get(key, 0)

def merge_dictionaries(d1, d2):
    '''
    INPUT: dictionary, dictionary
    OUTPUT: dictionary

    Return a new dictionary which contains all the keys from d1 and d2 with
    their associated values. If a key is in both dictionaries, the value should
    be the sum of the two values.
    '''
    d = d1.copy()
    for key, value in d2.items():
        d[key] = d.get(key, 0) + value
    return d



----------------------------------------------------
#decision tree

import numpy as np
import pandas as pd
import scipy
import scipy.stats as scs
import operator
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, tree
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

%matplotlib inline
# Make it pretty
plt.style.use('ggplot')

# Seed random functions for reproducibility
np.random.seed(3)

--------------------------------------
#parameters unpacked in a dictionary

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

# staged_predict
# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

--------------------------------------------
#pandas 

#want to change '567-987' to 987 (int)
def get_max_score(fico_range):
    return int(df.partition('-')[2])

df['fico_scores'] = df['fico_scores'].apply(get_max_score)

--------------------------------------------


