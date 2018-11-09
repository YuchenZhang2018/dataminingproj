# calcaute the correlation between all the "rolled-up" attributes and the Churn value
# and then use Navie Bayesian to classify
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import chisquare
import matplotlib.pyplot as plt

df = pd.read_csv('data/test.csv')

def calChi2(col_name):
	df_1 = df.filter([col_name,'Churn'], axis=1)
	df_4 =pd.crosstab(df_1[col_name], df_1.Churn)
	#print('the numberic distribution table\n', df_4)
	(x,_,_,_) = st.chi2_contingency(pd.crosstab(df_1[col_name], df_1.Churn))
	print('chi2:', x, 'for', col_name)


all_column = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 
				'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
				'StreamingTV', 'StreamingMovies', 'Contract','PaperlessBilling', 'PaymentMethod']#,'MonthlyCharges','TotalCharges']

#map(lambda x:x.calChi2(x),all_column)

for obj in all_column:
       calChi2(obj)




