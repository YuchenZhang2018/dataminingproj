from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
import scikitplot



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
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    precise = (tn+tp)/float(tn+fp+fn+tp)
    recall = tp/float(tp+fn)

def randomforest(n_estimators_=10, max_depth_=9,max_features_=9, min_samples_split_=10):
    forest = RandomForestClassifier(n_estimators=n_estimators_c, max_depth=max_depth_, max_features=max_features_, min_samples_split=min_samples_split_, bootstrap=True,
                                    n_jobs=3)
    return forest


n_estimatorslist=[3,5,10,15,20,30]
max_depthlist=[3,5,7,8,9,10,15]
max_featureslist = [5,7,9,10,12,15]
min_samples_splitlist=[5,10,15,30,50,100]
acc_estimators  =[]
acc_maxdepth=[]
acc_maxfeatures=[]
acc_min_samples_split=[]

train_X, train_y = processData(train_set)
test_X, test_y = processData(test_set)



for n in n_estimatorslist:
    forest = randomforest(n_estimators_=n)
    forest.fit(train_X, train_y)
    yhat = forest.predict(test_X)
    acc_estimators.append(accuracy(yhat,test_y))







print()

