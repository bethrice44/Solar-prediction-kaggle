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


def PCA(data):
    """Run principal component analysis on the data"""
    
    
    return data


def Train(X, Y, model, N):
    """Train the data on train-test-splits from the training data. Print the mean absolute error"""
    
    MAEs = 0
    for i in range(N):
        trainX, X_CV, trainY, Y_CV = train_test_split(X, Y)
        model.fit(trainX, trainY)
        predictions = model.predict(X_CV)
        predictions = np.clip(predictions, np.min(trainY), np.max(trainY))
        mae = MAE(Y_CV,predictions)
        MAEs += mae
    
    return MAEs/N


def MAE(predictions,target):
	''' Find the mean absolute error '''
	return np.mean(np.absolute(predictions-target))


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

	print "Defining model"

	model=RandomForestRegressor()
    
	print "Run CV loop on train-test-splits"

	Av_error=Train(TrainX_all,TrainY_all,model,20)

	print Av_error


if __name__ == "__main__":
	main()


