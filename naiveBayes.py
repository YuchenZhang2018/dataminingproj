# https://scikit-plot.readthedocs.io/en/stable/metrics.html

from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score

# create data frame containing your data, each column can be accessed # by df['column   name']
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df_all = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print('shape of the whole file', df_all.shape)
print('shape of train', df_train.shape)
print('shape of test', df_test.shape)
print ('in train', df_train['Churn'].value_counts())
print ('in test', df_test['Churn'].value_counts())

def countFre():
    train_1 = df_train.Churn.groupby(1).count()
    train_0 = df_train.Churn.groupby(0).count()
    test_1 = df_test.Churn.groupby(1).count()
    test_0 = df_test.Churn.groupby(0).count()



    print('1: ', train_1+ test_1)
    print('0: ', train_0+ test_0)



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

train_X, train_y = processData(df_train)
test_X, test_y = processData(df_test)



def drawConfusionMatrix(y_gnb, test_y):
    skplt.metrics.plot_confusion_matrix(y_gnb, test_y, normalize=True)
    plt.show()



def drawPRGraph(train_X, train_y, test_X, test_y, probas):
    skplt.metrics.plot_precision_recall(test_y, probas)
    plt.show()


def drawRoc(test_y, probas):
    skplt.metrics.plot_roc(test_y, probas)
    plt.show()


def calScores(test_y, y_gnb, probas):
    precision, recall, thresholds = skplt.metrics.precision_recall_curve(test_y, probas)
    # calculate F1 score
    f1 = f1_score(test_y, y_gnb)
    # calculate precision-recall AUC
    aucc = auc(recall, precision) # accuracy 1
    # calculate average precision score
    ap = average_precision_score(test_y, probas)
    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, aucc, ap))


#countFre()
gnb = GaussianNB()
y_gnb = gnb.fit(train_X, train_y).predict(test_X)

nb=gnb.fit(train_X, train_y)
probas = nb.predict_proba(test_X)
print('probs shape ', probas.shape)
# keep probabilities for the positive outcome only
probas = probas[:, 1]
print('probs shape 1 ', probas.shape)

accu = gnb.score(test_X, test_y) # accuracy 2
print(accu)
"""
drawConfusionMatrix(y_gnb, test_y)
drawPRGraph(train_X, train_y, test_X, test_y, probas)
drawRoc(test_y, probas)
"""
print(test_y.shape)
print('prob shape ',probas.shape)
calScores(test_y, y_gnb, probas)


