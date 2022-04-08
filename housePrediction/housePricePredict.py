import pandas as pd 
pd.options.display.max_columns = None
pd.options.display.max_rows = None
from housePredFunction import *
import numpy as np


input_string = input("Enter feature of house:")
feature_list = input_string.split(",")
feature_list=convertNanToNone(feature_list)
#feature_list  =[20, 'RL', 80.0, 9600, 'Pave', None, 'Reg', 'Lvl', 'AllPub', 'FR2', 'Gtl', 'Veenker', 'Feedr', 'Norm', '1Fam', '1Story', 6, 8, 1976, 1976, 'Gable', 'CompShg', 'MetalSd', 'MetalSd', 'None', 0.0, 'TA', 'TA', 'CBlock', 'Gd', 'TA', 'Gd', 'ALQ', 978, 'Unf', 0, 284, 1262, 'GasA', 'Ex', 'Y', 'SBrkr', 1262, 0, 0, 1262, 0, 1, 2, 0, 3, 1, 'TA', 6, 'Typ', 1, 'TA', 'Attchd', 1976.0, 'RFn', 2, 460, 'TA', 'TA', 'Y', 298, 0, 0, 0, 0, 0, None, None, None, 0, 5, 2007, 'WD', 'Normal']
feature_list=convertTypeInputList(feature_list)
#feature_list = (np.asarray(feature_list).reshape(1,79))
#print(feature_list)
test = getEmptyDataFrame()
test.loc[len(test.columns)]=feature_list
categorical_features = findCategoricalFeatures(test)
print(categorical_features)
numerical_features   = findNumericFeatures(test)
#print(numerical_features)
#print(len(categorical_features))
#print(len(numerical_features))
test = fillMissingNumericValues(test,numerical_features)
test = fillMissingCategoricalValues(test,categorical_features)
test= scaleWithPickle(test,numerical_features)
test = encodeWithPickle(test,categorical_features)
with open('xgboost_model.pickle', 'rb') as handle:
        xgb_model = pickle.load(handle)

pred = xgb_model.predict(test)
pred = inverseScaledTarget(pred)

print("Price of house:",pred)


#[20, 'RL', 80.0, 9600, 'Pave', nan, 'Reg', 'Lvl', 'AllPub', 'FR2', 'Gtl', 'Veenker', 'Feedr', 'Norm', '1Fam', '1Story', 6, 8, 1976, 1976, 'Gable', 'CompShg', 'MetalSd', 'MetalSd', 'None', 0.0, 'TA', 'TA', 'CBlock', 'Gd', 'TA', 'Gd', 'ALQ', 978, 'Unf', 0, 284, 1262, 'GasA', 'Ex', 'Y', 'SBrkr', 1262, 0, 0, 1262, 0, 1, 2, 0, 3, 1, 'TA', 6, 'Typ', 1, 'TA', 'Attchd', 1976.0, 'RFn', 2, 460, 'TA', 'TA', 'Y', 298, 0, 0, 0, 0, 0, nan, nan, nan, 0, 5, 2007, 'WD', 'Normal']
#20,RL,80.0,9600,Pave,None,Reg,Lvl,AllPub,FR2,Gtl,Veenker,Feedr,Norm,1Fam,1Story,6,8,1976,1976,Gable,CompShg,MetalSd,MetalSd,None,0.0,TA,TA,CBlock,Gd,TA,Gd,ALQ,978,Unf,0,284,1262,GasA,Ex,Y,SBrkr,1262,0,0,1262,0,1,2,0,3,1,TA,6,Typ,1,TA,Attchd,1976.0,RFn,2,460,TA,TA,Y,298,0,0,0,0,0,None,None,None,0,5,2007,WD,Normal 
#181500
#60,RL,84,14260,Pave,NA,IR1,Lvl,AllPub,FR2,Gtl,NoRidge,Norm,Norm,1Fam,2Story,8,5,2000,2000,Gable,CompShg,VinylSd,VinylSd,BrkFace,350,Gd,TA,PConc,Gd,TA,Av,GLQ,655,Unf,0,490,1145,GasA,Ex,Y,SBrkr,1145,1053,0,2198,1,0,2,1,4,1,Gd,9,Typ,1,TA,Attchd,2000,RFn,3,836,TA,TA,Y,192,84,0,0,0,0,NA,NA,NA,0,12,2008,WD,Normal
#250000



"""
test_df = pd.read_csv("test.csv")

test_df = dropColumn(test_df,'Id')

#a = np.asarray(test_df)

#print(a[1])
#print(len(a[1]))

categorical_features = findCategoricalFeatures(test_df)

numerical_features   = findNumericFeatures(test_df)

#print(numerical_features)
test_df = fillMissingNumericValues(test_df,numerical_features)
test_df = fillMissingCategoricalValues(test_df,categorical_features)
test_df=  scaleWithPickle(test_df,numerical_features)
test_df = encodeWithPickle(test_df,categorical_features)

with open('xgboost_model.pickle', 'rb') as handle:
        xgb_model = pickle.load(handle)

pred = xgb_model.predict(test_df)
pred = inverseScaledTarget(pred)

pred_frame = pd.DataFrame(pred,columns=['Predict'])
pred_frame.reset_index(drop=True,inplace=True)
print(pred_frame)

""" 



