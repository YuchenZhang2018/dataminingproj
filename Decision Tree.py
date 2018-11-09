import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('data/train.csv', index_col=0 )
df.head()
df.shape
#去掉Churn， customerID两列
trainx = df.drop(['Churn', 'customerID'], axis=1)
#得到label
trainy = df['Churn']
#TotalCharges列转换为float
trainx['TotalCharges']  = pd.to_numeric(trainx['TotalCharges'], errors='coerce')
trainx.dtypes
# 网格搜索，寻找最优参数
from xgboost import XGBClassifier
from sklearn import model_selection
from scipy import stats

clf = model_selection.RandomizedSearchCV(
    estimator=XGBClassifier(learning_rate=0.01),
    param_distributions={
        'n_estimators': stats.randint(600, 900),
        'subsample': stats.uniform(0.3, 0.7),
        'max_depth': stats.randint(2,10),
        'colsample_bytree': stats.uniform(0.5, 0.4),
        'min_child_weight': stats.uniform(0, 20),
        'gamma': stats.uniform(0,10),
        'reg_alpha': stats.uniform(0, 10),
        'reg_lambda': stats.uniform(0, 10),
    },
    n_iter=50,
    cv=3,
    scoring='roc_auc',
    random_state=42,
)
clf.fit(trainx, trainy)

from sklearn import metrics


# Combine all relevant outputs into one function
def print_performance(y_true, y_pred):
    print(metrics.classification_report(y_true, y_pred))
    print('Accuracy: {}'.format(metrics.accuracy_score(y_true, y_pred)))
    print('ROC AUC: {}'.format(metrics.roc_auc_score(y_true, y_pred)))
dftest = pd.read_csv('data/test.csv', index_col=0 )
testx = dftest.drop(['Churn', 'customerID'], axis=1)
testy = dftest['Churn']
testx['TotalCharges']  = pd.to_numeric(testx['TotalCharges'], errors='coerce')
y_test_pred = clf.predict(testx)
print("\n Performance on test Set")
print_performance(testy, y_test_pred)
#print clf.best_score_
#print clf.best_params_
#根据网格搜索得到的参数，重新得到模型，打印特征重要性
import  xgboost

para = {'reg_alpha': 0.3142918568673425, 'colsample_bytree': 0.6975182385457563, 'min_child_weight': 2.2178164162366265,
 'n_estimators': 778, 'subsample': 0.5200491867534287, 'reg_lambda': 6.364104112637804, 'max_depth': 8, 'gamma': 5.227328293819941}
model = xgboost.XGBClassifier(**para)
model.fit(trainx, trainy)

tyr = model.predict(testx)
print("\n Performance on test Set")
print_performance(testy, tyr)
### plot feature importance
from xgboost import plot_importance
fig,ax = plt.subplots(figsize=(15,15))
plot_importance(model,
                height=0.5,
                ax=ax,
                max_num_features=64)
plt.show()
