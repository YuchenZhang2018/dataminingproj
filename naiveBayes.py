from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
import numpy as np

# create data frame containing your data, each column can be accessed # by df['column   name']
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')


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


def drawConfusionMatrix(y_gnb, train_y):
    skplt.metrics.plot_confusion_matrix(y_gnb, train_y, normalize=True)
    plt.show()



def drawPRGraph(train_X, train_y, test_X, test_y, probas):
    skplt.metrics.plot_precision_recall(test_y, probas)
    plt.show()

gnb = GaussianNB()
y_gnb = gnb.fit(train_X, train_y).predict(test_X)

nb=gnb.fit(train_X, train_y)
probas = nb.predict_proba(test_X)

accu = gnb.score(test_X, test_y)
print(accu)

drawConfusionMatrix(y_gnb, train_y)
drawPRGraph(train_X, train_y, test_X, test_y, probas)


