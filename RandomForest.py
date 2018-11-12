from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt



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

# def accuracy(yhat, y):
#     tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
#     precise = (tn+tp)/float(tn+fp+fn+tp)
#     recall = tp/float(tp+fn)
#     return precise

def randomforest(n_estimators_=10, max_depth_=9,max_features_=9, min_samples_split_=10):
    forest = RandomForestClassifier(n_estimators=n_estimators_, max_depth=max_depth_, max_features=max_features_, min_samples_split=min_samples_split_, bootstrap=True,
                                    n_jobs=3)
    return forest

# skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)


n_estimatorslist=[3,5,10,15,30]
max_depthlist=[3,5,7,10,15]
max_featureslist = [5,7,9,10,12]
min_samples_splitlist=[5,10,15,50,100]
acc_estimators  =[]
acc_maxdepth=[]
acc_maxfeatures=[]
acc_min_samples_split=[]

train_X, train_y = processData(train_set)
test_X, test_y = processData(test_set)



for n in n_estimatorslist:
    forest = randomforest(n_estimators_=n)
    forest.fit(train_X, train_y)
    yhat = forest.predict_proba(test_X)
    # acc_estimators.append(accuracy(yhat,test_y))
    skplt.metrics.plot_roc(test_y, yhat)
    plt.title("n_estimators="+str(n))
    plt.savefig("n_estimator_"+str(n)+".jpg")

for n in max_depthlist:
    forest = randomforest(max_depth_=n)
    forest.fit(train_X, train_y)
    yhat = forest.predict_proba(test_X)
    # acc_estimators.append(accuracy(yhat,test_y))
    skplt.metrics.plot_roc(test_y, yhat)
    plt.title("max_depth=" + str(n))
    plt.savefig("max_depth_" + str(n) + ".jpg")

for n in max_featureslist:
    forest = randomforest(max_features_=n)
    forest.fit(train_X, train_y)
    yhat = forest.predict_proba(test_X)
    # acc_estimators.append(accuracy(yhat,test_y))
    skplt.metrics.plot_roc(test_y, yhat)
    plt.title("max_features=" + str(n))
    plt.savefig("max_features_" + str(n) + ".jpg")

for n in min_samples_splitlist:
    forest = randomforest(min_samples_split_=n)
    forest.fit(train_X, train_y)
    yhat = forest.predict_proba(test_X)
    # acc_estimators.append(accuracy(yhat,test_y))
    skplt.metrics.plot_roc(test_y, yhat)
    plt.title("min_samples_split="+str(n))
    plt.savefig("min_samples_split_"+str(n)+".jpg")


forest = randomforest(max_features_=10, n_estimators_=15,max_depth_=10)
forest.fit(train_X, train_y)
yhat = forest.predict_proba(test_X)
skplt.metrics.plot_roc(test_y, yhat)
plt.title("ROC Curve")
plt.savefig("final ROC Curve")

skplt.metrics.plot_precision_recall(test_y, yhat)
plt.savefig("final Precise-Recall Curve")

skplt.metrics.plot_precision_recall(test_y, yhat)
plt.savefig("final Precise-Recall Curve")

# skplt.metrics.plot_confusion_matrix(test_y, yhat, normalize=True)
# plt.savefig("final Confusion Matrix table")