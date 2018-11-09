from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

# create data frame containing your data, each column can be accessed # by df['column   name']
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')


def processData(data):
    data = data.drop('Unnamed: 0',1)
    data = data.drop('customerID',1)
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
    data_arr = data.values
    length = len(data_arr[0])
    for i in range(len(data_arr)):
        if data_arr[i][length-1] ==' ':
            data_arr[i][length - 1] = 0
            continue;
        data_arr[i][length-1] = float(data_arr[i][length-1])

    return data_arr, label

train_X, train_y = processData(df_train)
test_X, test_y = processData(df_test)

def accuracy(yhat, y):
    dif = yhat-y
    count = 0
    for item in dif:
        if item==0:
            count+=1
    return count/float(len(y))


gnb = GaussianNB()
y_gnb = gnb.fit(train_X, train_y).predict(test_X)
print(accuracy(y_gnb,test_y))
