# clean the data
# separate test and train with their features

from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np



def cleanNonParsable(df):
	df = df.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
	return df


def processData(data):
    data = data.drop('Unnamed: 0',1)
    data = data.drop('customerID',1)
    # get the feature column out
    label = data['Churn']
    data = data.drop('Churn',1)
    data['PhoneService'] = data['PhoneService']+data['MultipleLines']
    data = data.drop('MultipleLines',1)
    data['InternetService'] = data['InternetService'] + data['OnlineSecurity']+data['OnlineBackup'] \
                              +data['DeviceProtection']+ data['TechSupport']+data['StreamingTV'] + data['StreamingMovies']
    data = data.drop('OnlineSecurity',1)
    data = data.drop('OnlineBackup',1)
    data = data.drop('DeviceProtection',1)
    data = data.drop('TechSupport',1)
    data = data.drop('StreamingTV',1)
    data = data.drop('StreamingMovies',1)

    # change from data frame to Numpy presentation
    data_arr = data.values
    # continue to clean the non convertabile 'space' value to numeric zero
    # this happened in which column??
    length = len(data_arr[0])
    for i in range(len(data_arr)):
        if data_arr[i][length-1] ==' ':
            data_arr[i][length - 1] = 0
            continue;
        data_arr[i][length-1] = float(data_arr[i][length-1])


# df.to_csv(file_name, sep='\t')
    return data_arr, label


# create data frame containing your data, each column can be accessed # by df['column   name']
df= pd.read_csv('data/test_01.csv')

print(df.columns)
# give first column a name 
#df.rename({'Unnamed: 0' : 'order'}, axis='columns', inplace= True)
# then ignore the order column when processing
#df.drop('order',axis=1,inplace=True)
df.drop('Unnamed: 0',axis=1,inplace=True)
#df.rename({'Unnamed: 0' : 'order'}, axis='columns', inplace= True)
#df_test = pd.read_csv('data/test.csv')
df_cleaned = cleanNonParsable(df)
#df_cleaned.drop(, axis=1,inplace=True)
df_cleaned.to_csv('data/test_after_clean.csv')



#train_X, train_y = processData(df_train)
#test_X, test_y = processData(df_test)
