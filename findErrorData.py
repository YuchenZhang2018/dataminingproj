# clean the data
# separate test and train with their features

from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np



def cleanNonParsable(df):
	df = df.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
	return df

# create data frame containing your data, each column can be accessed # by df['column   name']

def genCleanedDataFile(filename):
    df= pd.read_csv('data/'+filename)

    df.drop('Unnamed: 0',axis=1,inplace=True)

    df_cleaned = cleanNonParsable(df)

    # aggregation columns (sum and delete)

    df_cleaned['InternetServiceTotal']=df_cleaned.iloc[:,-13:-6].sum(axis = 1) # add up 7 columns into 1
    #print(df_cleaned['InternetServiceTotal'])


    df_cleaned.drop(df_cleaned.iloc[:, -14:-7], axis = 1, inplace=True)

    df_cleaned.to_csv('data_cleaned/'+filename)


genCleanedDataFile('Telco_Churn.csv')
