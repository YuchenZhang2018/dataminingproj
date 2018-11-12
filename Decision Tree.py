import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('data/train.csv', index_col=0 )

print df.shape
df.head()
df.shape

#df = df.fillna('0')
trainx = df.drop(['Churn', 'customerID'], axis=1)

trainy = df['Churn']

trainx['TotalCharges']  = pd.to_numeric(trainx['TotalCharges'], errors='coerce')
trainx = trainx.fillna('0')
trainx.shape

from scipy import stats
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainx, trainy)
from sklearn import metrics

# Combine all relevant outputs into one function
def print_performance(y_true, y_pred):
    display(pd.DataFrame(metrics.confusion_matrix(y_true, y_pred)))
    print(metrics.classification_report(y_true, y_pred))
    print('Accuracy: {}'.format(metrics.accuracy_score(y_true, y_pred)))
    print('ROC AUC: {}'.format(metrics.roc_auc_score(y_true, y_pred)))
    dftest = pd.read_csv('test.csv', index_col=0)
    testx = dftest.drop(['Churn', 'customerID'], axis=1)
    testx['TotalCharges'] = pd.to_numeric(testx['TotalCharges'], errors='coerce')
    testx = testx.fillna(0)
    testy = dftest['Churn']

    y_test_pred = clf.predict(testx)
    print("\n Performance on test Set")
    print_performance(testy, y_test_pred)
    # plt.hist(testy)

    ### plot feature importance
    # 值越大，越重要
    print clf.feature_importances_

