# calcaute the correlation between all the "rolled-up" attributes and the Churn value
# and then use Navie Bayesian to classify


# need to run after cleaned/ processed
# run for combined of test and train
# for individual features and aggreated features


import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import chisquare
import matplotlib.pyplot as plt
from operator import itemgetter
from findErrorData import genCleanedDataFile

df = pd.read_csv('data/test.csv')

def calChi2(df, col_name):
	df_1 = df.filter([col_name,'Churn'], axis=1)
	df_4 =pd.crosstab(df_1[col_name], df_1.Churn)
	#print('the numberic distribution table\n', df_4)
	(chi2,_,_,_) = st.chi2_contingency(pd.crosstab(df_1[col_name], df_1.Churn))
	#print('chi2:', chi2, 'for', col_name)
	return chi2



def sortChiList(df, col_list):
	lst = []
	for obj in col_list:
	       chi2 = calChi2(df, obj)
	       lst.append((obj,chi2))

	lst.sort(key=itemgetter(1), reverse = True)
	print(lst)

# before combining the 7 columnns
all_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 
				'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
				'StreamingTV', 'StreamingMovies', 'Contract','PaperlessBilling', 'PaymentMethod']#,'MonthlyCharges','TotalCharges']

sortChiList(df, all_columns)

# get combined data
genCleanedDataFile('Telco_Churn.csv')
# get all column names in a list
df_all = pd.read_csv('data_cleaned/Telco_Churn.csv')
#all_columns_2 = df_all.columns.get_values().tolist()

# after combing the 7 columns

all_columns_3 = [ 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
				'MultipleLines', 'Contract', 'PaperlessBilling', 'PaymentMethod',  
				'InternetServiceTotal'] # 'MonthlyCharges', 'TotalCharges',

sortChiList(df_all, all_columns_3)


