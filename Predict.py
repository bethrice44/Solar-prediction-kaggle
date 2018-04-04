#!/usr/bin/env python

"""
AMS Solar Energy Prediction

"""

import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import os
import sklearn as sk
from netCDF4 import Dataset


def import_csv_data():
	
	df_train=pd.read_csv('train.csv')
	df_test=pd.read_csv('test.csv')
	df_stations=pd.read_csv('station_info.csv')

	return df_train,df_test,df_stations


def split_df(df_data):
    
    times=df_data[:,0].astype(int)
    data=[:,1:]
    
    return times,data


def get_all_predictors(predictors):
	
	for i,predictor in enumerate(predictors):
		if i==0:
			X=get_predictor(predictor)
		else:
			X_append=get_predictor(predictor)
			X=np.hstack((X,X_append))

	return X


def get_predictor(predictor):

	X=Dataset(os.path.join(path,i+train_end)).variables.values()[-1][:]
	X=X.reshape(X.shape[0],55,9,16)
	X=np.mean(X,axis=1)
	X=X.reshape(X.shape[0],np.prod(X.shape[1:]))

	return X




def make_stationary():
    

    return X



def train_model():

    return model



def predict():

    return prediction


def MAE(predictions,target):
	''' Find the mean absolute error '''
	return np.mean(np.absolute(predictions-target))


def main():
	
	predictors = ['apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm',\
                  'spfh_2m','tcdc_eatm','tcolc_eatm','tmax_2m','tmin_2m',\
                  'tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sfc']

	train_end = '_latlon_subset_19940101_20071231.nc'
	train_path='train/'

	test_end = '_latlon_subset_20080101_20121130.nc'
	test_path='test/'

    model=sk.Ridge(normalize=True)



if __name__ == "__main__":
	main()


