from sklearn.ensemble import RandomForestClassifier
import pandas as pd

forest = RandomForestClassifier(n_estimators=10, max_depth=9,max_features=9, min_samples_split=10, bootstrap = True, n_jobs=3)

train_set = pd.read_csv('data/train.csv')
test_set = pd.read_csv('data/test.csv')

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

def accuracy(yhat, y):
    dif = yhat-y
    count = 0
    for item in dif:
        if item==0:
            count+=1
    return count/float(len(y))


train_X, train_y = processData(train_set)
test_X, test_y = processData(test_set)

forest.fit(train_X,train_y)
yhat = forest.predict(test_X)
print(accuracy(yhat,test_y))

