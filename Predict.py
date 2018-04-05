#!/usr/bin/env python

"""
AMS Solar Energy Prediction

"""

import pandas as pd
import numpy as np
import os
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from netCDF4 import Dataset


def import_csv_data():
	"""Import csv training data containing the total daily incoming solar energy in (J m-2) at 98 Oklahoma Mesonet sites"""
	df_train=np.loadtxt('train.csv',delimiter=',',dtype=float,skiprows=1)

	return df_train


def split_times(df_data):
    """Split so datetime is separate from the solar data"""
    
    times=df_data[:,0].astype(int)
    data=df_data[:,1:]
    
    return times,data


def get_all_predictors(path,predictors,postfix):
    """Get all the predicting data for train and test"""
	
	for i,predictor in enumerate(predictors):
		if i==0:
			X=get_predictor(path,predictor,postfix)
		else:
			X_append=get_predictor(path,predictor,postfix)
			X=np.hstack((X,X_append))

	return X


def get_predictor(path,predictor,postfix):
    """Get predicting data for train and test for a sepcific predictor"""

	X=Dataset(os.path.join(path,predictor+postfix)).variables.values()[-1][:]
	X=X.reshape(X.shape[0],55,9,16)
	X=np.mean(X,axis=1)
	X=X.reshape(X.shape[0],np.prod(X.shape[1:]))

	return X


def Run_random_forest(trainX,trainY,testX):
    """Run a random forest regressor model"""
    
    init_model=ensemble.RandomForestRegressor()
    parameters={'n_estimators':np.linspace(5, trainX.shape[1], 20).astype(int)}
    gridCV=grid_search.GridSearchCV(init_model,parameters,cv=10)
    
    trainX_split,testX_split,trainY_split,testY_split = train_test_split(trainX,trainY,test_size=500)
    gridCV.fit(trainX_split,trainY_split)
    
    n_estimators = gridCV.best_params_['n_estimators']
    print n_estimators
    
    model=ensemble.RandomForestRegressor(n_estimators=n_estimators)
    
    print "Fitting model..."
    
    model.fit(trainX,trainY)
    predictions=model.predict(testX)
    
    return predictions


def Run_SVR(trainX,trainY,testX):
    """Run a support vector regression model"""
    
    init_model=svm.SVR()
    parameters={'C':np.logspace(-5,5,10), 'gamma':np.logspace(-5,5,10), 'epsilon':np.logspace(-2,2,10)}
    gridCV=grid_search.GridSearchCV(init_model,parameters,cv=10)
    
    trainX_split,testX_split,trainY_split,testY_split = train_test_split(trainX,trainY,test_size=500)
    gridCV.fit(trainX_split,trainY_split)
    
    gamma = gridCV.best_params_['gamma']
    C = gridCV.best_params_['C']
    epsilon = gridCV.best_params_['epsilon']
    
    print gamma, C, epsilon
    
    model=svm.SVR(C=C, gamma=gamma, epsilon=epsilon)
    
    print "Fitting model..."
    
    model.fit(trainX,trainY)
    predictions=model.predict(testX)
    
    return predictions


def Run_ridge(trainX,trainY,testX):
    """Run a Ridge model"""
    
    model=linear_model.RidgeCV(alphas=np.logspace(-0,3,100),cv=5)
    
    print "Fitting model..."
    
    model.fit(trainX,trainY)
    predictions=model.predict(testX)
    
    return predictions


def Run_GBR(trainX,trainY,testX):
    """Run a Gradient Bosted Regressor model"""
    
    parameters={"loss": "lad", "n_estimators": 3000, "learning_rate": 0.035, "max_features": 80,  "max_depth": 7, "subsample": 0.5}
    
    model=ensemble.GradientBoostingRegressor(parameters)
    
    print "Fitting model..."
    
    model.fit(trainX,trainY)
    predictions=model.predict(testX)
    
    return predictions


def main():
    
    """Using all predictors"""
    
	Predictors = ['apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm',\
                  'spfh_2m','tcdc_eatm','tcolc_eatm','tmax_2m','tmin_2m',\
                  'tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sfc']

	train_end = '_latlon_subset_19940101_20071231.nc'
	train_path='train/'

	test_end = '_latlon_subset_20080101_20121130.nc'
	test_path='test/'

	print "Importing trainX, testX..."

	TrainX_all=get_all_predictors(train_path,Predictors,train_end)
	TestX_all=get_all_predictors(test_path,Predictors,test_end)
    
	print "Importing trainY..."
        
	df_Train=import_csv_data()

	Times,TrainY_all=split_times(df_Train)

    Predictions_RF=Run_random_forest(TrainX_all,TrainY_all,testX_all)
    
    Predictions_SVR=Run_SVR(TrainX_all,TrainY_all,testX_all)
    
    Predictions_Ridge=Run_ridge(TrainX_all,TrainY_all,testX_all)

    Predictions_GBR=Run_GBR()


if __name__ == "__main__":
	main()


