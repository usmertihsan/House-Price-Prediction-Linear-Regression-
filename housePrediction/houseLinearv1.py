import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression,LassoCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
pd.options.display.max_columns = None
pd.options.display.max_rows = None
from housePredFunction import *
import pickle

# import train and test data
train_df = pd.read_csv("train.csv")

# print shape of datas
print("\nThe train data size:{} ".format(train_df.shape)) 

#drop specific column



train_df = dropColumn(train_df,"Id")

train_df
#find numeric and categorical feature names
print(train_df.dtypes)
categorical_features = findCategoricalFeatures(train_df)
    
numerical_features   = findNumericFeatures(train_df)


# detect outliers and get rid of them

train_df  = getRidOfOutliers(train_df,numerical_features)

#fill missing values 

train_df = fillMissingNumericValues(train_df,numerical_features)
train_df = fillMissingCategoricalValues(train_df,categorical_features)

# scale numeric values 
train_df=scalerTarget(train_df,'SalePrice')

numerical_features=removeTarget(numerical_features,'SalePrice')

train_df= scalerNumericFeatures(train_df,numerical_features)


print(len(categorical_features))
print(len(numerical_features))

#train_df = inverseScaledNumericFeatures(train_df,numerical_features)

#train_df.head(5)



# Encode categorical variables

train_df = encodeCategoricalFeatures(train_df,categorical_features)


train_df


# train and test split for model train

X_train = train_df.drop(labels = ["SalePrice"], axis = 1)
y_train = train_df["SalePrice"].values

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
""" 
linear_reg = LinearRegression()
linear_reg.fit(X_train,y_train)
y_head = linear_reg.predict(X_test)
print("\n MULTIPLE LINEAR REGRESSION SCORES:")
print('\nR square Accuracy: ',r2_score(y_test,y_head))
print('\nMean Absolute Error Accuracy: ',mean_absolute_error(y_test,y_head))
print('\nMean Squared Error Accuracy: ',mean_squared_error(y_test,y_head))
"""
model = LassoCV(cv=5, random_state=0, max_iter=10000)

model.fit(X_train, y_train)

best_alpha = model.alpha_

lasso_best = Lasso(alpha=best_alpha)
lasso_best.fit(X_train, y_train)
print("\n LASSO SCORES:")
print('\nBest alpha:',model.alpha_)
print('\nR squared training set', round(lasso_best.score(X_train, y_train)*100, 2))
print('\nR squared test set', round(lasso_best.score(X_test, y_test)*100, 2))

#n_estimators = number of trees in the foreset
#max_features = max number of features considered for splitting a node
#max_depth = max number of levels in each decision tree
#min_samples_split = min number of data points placed in a node before the node is split
#min_samples_leaf = min number of data points allowed in a leaf node
#bootstrap = method for sampling data points (with or without replacement)

#hyper parameter --> max_dept=2 de 0.81 r^2, max_dept=3 de 0.85 r^2, max_dept=4 de 0.87 r^2,max_dept=5 de 0.88 r^2,
# max_dept=6 de 0.88 r^2, the others same result, the best one max_dept=5
""" 
regr = RandomForestRegressor(random_state = 42,bootstrap = True, max_depth=5,max_features='auto',min_samples_leaf=1,min_samples_split=2,n_estimators=10)
pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('reduce_dim', PCA()),
        ('regressor', regr)
        ])
pipe.fit(X_train,y_train)
ypipe=pipe.predict(X_test)

print("\nRANDOM FOREST PIPELINE SCORES:")
print('\nR square Accuracy: ',r2_score(y_test,ypipe))
print('\nMean Absolute Error Accuracy: ',mean_absolute_error(y_test,ypipe))
print('\nMean Squared Error Accuracy: ',mean_squared_error(y_test,ypipe))
rmse = float(format(np.sqrt(mean_squared_error(y_test, ypipe)), '.3f'))
print("\nRMSE: ", rmse)

"""



""" 
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize =(5,5))
sns.regplot(x=y_xgbpredict,y=y_test-y_xgbpredict,ax=ax,lowess=True)
ax.set(ylabel='residuals',xlabel='fitted values')
plt.show()
"""

xgboost_model = XGBRegressor(n_estimators=1200, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8)
xgboost_model.fit(X_train, y_train)
y_xgbpredict = xgboost_model.predict(X_test)


with open('xgboost_model.pickle', 'wb') as handle:
         pickle.dump(xgboost_model, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("\n XGB BOOST SCORES:")
print('\nR square Accuracy: ',r2_score(y_test,y_xgbpredict))
print('\nMean Absolute Error Accuracy: ',mean_absolute_error(y_test,y_xgbpredict))
rmseXgb = float(format(np.sqrt(mean_squared_error(y_test, y_xgbpredict)), '.3f'))
print("\nRMSE: ", rmseXgb)



y_test = inverseScaledTarget(y_test)
y_xgbpredict = inverseScaledTarget(y_xgbpredict)

y_xgbpredict = pd.DataFrame(y_xgbpredict,columns=['Predict'])
y_test = pd.DataFrame(y_test,columns=['Test'])
y_test.reset_index(drop=True,inplace=True)
ytest_head=pd.concat([y_test,y_xgbpredict],axis=1)



ytest_head

print(ytest_head)