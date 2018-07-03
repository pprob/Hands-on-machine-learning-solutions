# New script for real estate regression - model building

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the data
housing = pd.read_csv('housing.csv')

# Adding new, more specific attributes
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# Moving categorical variables to end of dataframe
cols = list(housing.columns.values)
housing = housing[['longitude',
                      'latitude',
                      'housing_median_age',
                      'total_rooms',
                      'total_bedrooms',
                      'population',
                      'households',
                      'median_income',
                      'median_house_value',
                      'rooms_per_household',
                      'bedrooms_per_room',
                      'population_per_household',
                      'ocean_proximity']]


# Correlation of variables to median house value
correlation_matrix = housing.corr()
correlation_matrix["median_house_value"].sort_values(ascending=False)



# Correlation of new attributes to median house value
correlation_matrix = housing.corr()
correlation_matrix["median_house_value"].sort_values(ascending=False)

# Dealing with missing data
housing.info() # missing data under
housing_numerical = housing.drop('ocean_proximity', axis = 1)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median')
X = imputer.fit(housing_numerical)
imputer.statistics_
housing_numerical.median().values #Medians match
X = imputer.transform(housing_numerical)
np.isnan(np.sum(housing_numerical)) # NaN check

# Encoding categorical data
housing['ocean_proximity'].value_counts() #5 Categories
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
housing_cat = housing['ocean_proximity']
labelencoder = LabelEncoder()
housing_cat = labelencoder.fit_transform(housing_cat)
housing_cat_reshaped = housing_cat.reshape(-1,1)
onehotencoder = OneHotEncoder()
housing_cat_1hot = onehotencoder.fit_transform(housing_cat_reshaped).toarray()

# combining encoded categorical data to numerical variables
X = np.concatenate((X, housing_cat_1hot), axis = 1)

# Splitting data into a training set and a test set

# To avoid stratum bias, must ensure enough data in each stratum. Therefore create new grouping to avoid stratum bias.
# Choose median as this has highest correlation to determining medium_house_value
X = pd.DataFrame(X)
X.columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households',
             'median_income', 'median_house_value', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household', 'a', 'b', 'c', 'd', 'e']
X["income_cat"] = np.ceil(housing["median_income"] / 1.5)
X["income_cat"].where(X["income_cat"] < 5, 5.0, inplace=True)

# Split training data into test set and training set - Stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(test_size = 0.2, random_state = 0)
for train_index, test_index in split.split(X, X['income_cat']):
    train_set = X.loc[train_index]
    test_set = X.loc[test_index]
#Checking proportionality of split. total data vs test split
X['income_cat'].value_counts()/len(X)
test_set["income_cat"].value_counts() / len(test_set)

# After splitting, drop income_cat
train_set = train_set.drop(['income_cat'], axis = 1)
test_set = test_set.drop(['income_cat'], axis = 1)

# Specify label for train and test set
y_train = train_set[['median_house_value']]
y_test = test_set[['median_house_value']]

# Specify features
X_train = train_set.drop(['median_house_value'], axis = 1)
X_test = test_set.drop(['median_house_value'], axis = 1)

# Feature scaling x_train and x_test - standardization
from sklearn.preprocessing import StandardScaler
standardscaler_X = StandardScaler()
X_train = standardscaler_X.fit_transform(X_train)
X_test = standardscaler_X.transform(X_test)

# Training and evaluating on the training set - OLS regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() # OLS
lin_reg.fit(X_train, y_train)

# Try out predictions on training set
y_pred_train = lin_reg.predict(X_train)

# Measure the RMSE - cost function
from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(y_true = y_train, y_pred = y_pred_train) 
lin_rmse = np.sqrt(lin_mse)
lin_rmse # prediction error of $67989 - underfitting data. Could regularize, or use more complex training model.

# implementing decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

# Decision tree regression predictions
y_pred_train_tree = tree_reg.predict(X_train)
tree_mse = mean_squared_error(y_true = y_train, y_pred = y_pred_train_tree)
tree_rmse = np.sqrt(tree_mse)
tree_rmse # Outputs 0 for the error. Clearly the model is overfitting the training data and may poorly generalize to future values.

# Implementing k-fold cross validation - split test set into 10 folds
from sklearn.model_selection import cross_val_score
tree_scores = cross_val_score(estimator = tree_reg, X = X_train, y = y_train, cv = 10, scoring = 'neg_mean_squared_error')
tree_rmse_scores = np.sqrt(-tree_scores)
tree_rmse_scores.mean()
tree_rmse_scores.std()

# k fold cross validation - lin reg, 10 folds
lin_scores = cross_val_score(estimator = lin_reg, X = X_train, y = y_train, cv = 10, scoring = 'neg_mean_squared_error')
lin_rmse_scores = np.sqrt(-lin_scores)
lin_rmse_scores.mean()
lin_rmse_scores.std()

# results: tree - mean = 71387+- 2528, lin = 68228+- 2584
# Linear regression model performs better. Decision tree model overfitting as well.

# Implementing random forest regression
from sklearn.ensemble import RandomForestRegressor
ran_forest_reg = RandomForestRegressor()
ran_forest_reg.fit(X_train, y_train.values.ravel()) #values.ravel() 1d array was expected, change shape of y 

forest_predict = ran_forest_reg.predict(X_train)
forest_mse = mean_squared_error(y_true = y_train, y_pred = forest_predict)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# k fold cross - random forest, 10 folds
ran_forest_scores = cross_val_score(estimator = ran_forest_reg, X = X_train, y = y_train.values.ravel(), cv = 10, scoring = 'neg_mean_squared_error')
ran_forest_rmse_scores = np.sqrt(-ran_forest_scores)
ran_forest_rmse_scores.mean()
ran_forest_rmse_scores.std()

# results validation mean = 52900+- 2182, by contrast results on training, mean = 22066
# training set, much lower than validation. Therefore still overfitting. Can regularize (constrain), or get more data

# Hyperparameter optimization with grid search
from sklearn.model_selection import GridSearchCV
parameters = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10, 30], 'max_features': [2, 3, 4, 5]},]

grid_search = GridSearchCV(estimator = ran_forest_reg, param_grid = parameters, scoring = 'neg_mean_squared_error', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
grid_search.best_estimator_


# using optimized hyperparameters
new_for_reg = RandomForestRegressor(
           bootstrap=False, criterion='mse', max_depth=None,
           max_features=5, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=30, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
new_for_reg.fit(X_train, y_train.values.ravel())

newfor_predict = new_for_reg.predict(X_train)
newfor_mse = mean_squared_error(y_true = y_train, y_pred = newfor_predict)
newfor_rmse = np.sqrt(newfor_mse)
newfor_rmse  # 0

newforest_scores = cross_val_score(estimator = new_for_reg, X = X_train, y = y_train.values.ravel(), cv = 10, scoring = 'neg_mean_squared_error')
newforest_rmse_scores = np.sqrt(-newforest_scores)
newforest_rmse_scores.mean() # 48958 +- 2060
newforest_rmse_scores.std()

# Predicting test set
y_test_pred = new_for_reg.predict(X_test)
test_mse = mean_squared_error(y_true = y_test, y_pred = y_test_pred)
test_mse
test_rmse = np.sqrt(test_mse)
test_rmse # = 47513







