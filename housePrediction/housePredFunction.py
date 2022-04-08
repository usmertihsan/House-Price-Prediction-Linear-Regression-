import sys
from unicodedata import category
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from collections import Counter
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
pd.options.display.max_columns = None
pd.options.display.max_rows = None
from sklearn.preprocessing import OrdinalEncoder

def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


def getRidOfOutliers(df,numerical_features):
    df = df.drop(detect_outliers(df,numerical_features),axis = 0).reset_index(drop = True)
    return df


def dropColumn(df,columnName):
    df = df.drop(columnName,axis=1)
    return df

def findCategoricalFeatures(df):  # find categorical feature names
   catCols = [col for col in df.columns if df[col].dtype=="O"]

   return catCols 
   
 # there is a problem in here, it just take names of not with datatypes
def getEmptyDataFrame():
   
    datatype = []
    test_df = pd.read_csv("test.csv")
    test_df = dropColumn(test_df,"Id")
    sendDataFrame = test_df.copy()
    sendDataFrame.drop(sendDataFrame.index,inplace=True)
    return sendDataFrame


def getColumns():
     columnName = []
     test_df = pd.read_csv("test.csv")
     test_df = dropColumn(test_df,"Id")
     for i in test_df.col:
         columnName.append(i)
     return columnName
 
 
def findNumericFeatures(df):  # find numeric feature names
    
  numeric = [] 
  datatype = df.dtypes
  numeric = datatype[(datatype == 'int64') | (datatype == 'float64')].index.tolist()
  return numeric 


def removeTarget(numeric,target):  
    numeric.remove(target)
    return numeric

def encodeCategoricalFeatures(df,categorical):
   label_object = {}
   for categoric in categorical:
       label_object[categoric]=LabelEncoder().fit(df[categoric])
       df[categoric] = label_object[categoric].transform(df[categoric]) 
 
   with open('labelEncodes.pickle', 'wb') as handle:
         pickle.dump(label_object, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    
   return df


def encodeWithPickle(df,categorical):
    with open('labelEncodes.pickle', 'rb') as handle:
        label_object = pickle.load(handle)
    for categoric in categorical:
        df[categoric] = label_object[categoric].transform(df[categoric])   
    return df


    
def scaleWithPickle(df,numerical):
    with open('scalerInp.pickle', 'rb') as handle:
        scaler = pickle.load(handle)
    df_orginal = df.copy()  
    df = df.select_dtypes(exclude=['object'])
    df = pd.DataFrame(scaler.transform(df.values), columns=df.columns, index=df.index)
    for numeric in numerical:
        df_orginal[numeric]=df[numeric]  
    return df_orginal    
 

 
def inverseEncodedCategoricalFeatures(df,categorical):
    with open('labelEncodes.pickle', 'rb') as handle:
        label_object = pickle.load(handle)
    for categoric in categorical:
         df[categoric] = label_object[categoric].inverse_transform(df[categoric])
         
    return df
    
def scalerNumericFeatures(df,numerical):
    
    df_orginal = df.copy()  
    df = df.select_dtypes(exclude=['object'])
    df=dropColumn(df,"SalePrice")
    
    scaler = MinMaxScaler(feature_range=(0, 1.5))
    df = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)
    
    for numeric in numerical:
        df_orginal[numeric]=df[numeric]
  
    with open('scalerInp.pickle', 'wb') as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    return df_orginal


    


def inverseScaledNumericFeatures(df,numerical):
    with open('scalerInp.pickle', 'rb') as handle:
        scaler = pickle.load(handle)
    df_orginal = df.copy()  
    df = pd.DataFrame(scaler.inverse_transform(df[numerical].values), columns=df[numerical].columns, index=df[numerical].index)
    for numeric in numerical:
        df_orginal[numeric]=df[numeric]  
    return df_orginal    
    
    


def scalerTarget(df,target):
    scaler = MinMaxScaler()
    df[target]= scaler.fit_transform(np.asarray(df[target]).reshape(-1,1))
    with open('scalerOut.pickle', 'wb') as handle:
         pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    return df
    



def inverseScaledTarget(target):
    with open('scalerOut.pickle', 'rb') as handle:
        scaler = pickle.load(handle)
    target_inversed= scaler.inverse_transform(np.asarray(target).reshape(-1,1))
    return target_inversed   

def scalerPredict(predict):
    scaler = MinMaxScaler()
    predict= scaler.fit_transform(np.asarray(predict).reshape(-1,1))
    with open('scalerPredictInp.pickle', 'wb') as handle:
         pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    return predict   

def inverseScaledPrediction(prediction):
    with open('scalerPredictInp.pickle', 'rb') as handle:
        scaler = pickle.load(handle)
    prediction_inversed= scaler.inverse_transform(np.asarray(prediction).reshape(-1,1))
    return prediction_inversed       

def fillMissingNumericValues(df,numerical):
    for numeric in numerical:
        df[numeric] = df[numeric].fillna(0)
    return df    
        
def convertNanToNone(listFeature):
    for i in range(len(listFeature)):
        if listFeature[i]=='NA':
            listFeature[i]=None
    return listFeature    
    
def fillMissingCategoricalValues(df,categorical):
    for categoric in categorical:
        df[categoric] = df[categoric].fillna("None")
    return df    


def convertTypeInputList(features):
    typeList = ['int64', 'object', 'float64','int64','object','object','object', 'object','object','object', 'object',  'object','object','object','object','object','int64','int64', 'int64','int64','object','object','object','object','object','float64','object','object','object','object','object','object','object','int64','object','int64','int64','int64','object','object','object','object','int64','int64','int64','int64','int64','int64','int64','int64','int64','int64','object','int64','object','int64','object','object','float64','object','int64','int64','object','object','object','int64','int64','int64','int64','int64','int64','object','object','object','int64','int64','int64','object','object' ]
    for i in range(len(features)):
        input = features[i]
        if typeList[i]=='int64':
            features[i]=int(input)
        elif typeList[i]=='float64':
            features[i]=float(input)
    return features

def check_user_input(input,feature,i):
    try:
        # Convert it into integer
        val = int(input)
        print("Input is an integer number. Number = ", val)
        feature[i] = int(input)
        return feature
    except ValueError:
        try:
            # Convert it into float
            val = float(input)
            print("Input is a float  number. Number = ", val)
            feature[i] = float(input)
            return feature 
          
        except ValueError:
            return feature
           
            


""" 
#print(categorical_features)

#print(numerical_features)

#print(len(categorical_features))

#print(len(numerical_features))

#train_df.isnull().any().sum()
#List of features which have missing values

## Filling missing values

def isEmptyFillNA(df,feature):
    df[feature] =df[feature].fillna("None")
   
def isZeroFillMean(df,feature):
    df[feature] = df[feature].fillna(df[feature].mean())   

def isEmptyFillModeZero(df,feature):
    df[feature] = df[feature].fillna(df[feature].mode()[0])

def isEmptyFillZero(df,feature):
    df[feature] = df[feature].fillna(0)
    
    
def fillSpecific(df,feature,choice):
    df[feature] = df[feature].fillna(choice)
     
"""
""" 
def scalerNumericFeatures(df,numerical):
   
    scaler = MinMaxScaler()
    for i in numerical:
        df[i]= scaler.fit_transform(np.asarray(df[i]).reshape(-1,1))
   
   
  
    with open('scalerInp.pickle', 'wb') as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    return df 

   
    
"""  