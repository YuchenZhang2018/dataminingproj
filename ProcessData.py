import pandas as pd

def transforForamt(dataset, diction):
    for i in range(len(dataset)):
        dataset[i] = diction[dataset[i]]
    return dataset


data_set = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

dic={"Female":1,"Male":0}
data_set['gender'] = transforForamt(data_set['gender'],dic)


dic_2={"Yes":1,"No":0}
data_set['Partner'] = transforForamt(data_set['Partner'],dic_2)
data_set['Dependents'] = transforForamt(data_set['Dependents'],dic_2)
data_set['PhoneService'] = transforForamt(data_set['PhoneService'],dic_2)
data_set['PaperlessBilling'] = transforForamt(data_set['PaperlessBilling'],dic_2)
data_set['Churn'] = transforForamt(data_set['Churn'],dic_2)

dic_3={"No internet service":-1,"Yes":1, "No":0}
data_set['OnlineSecurity'] = transforForamt(data_set['OnlineSecurity'],dic_3)
data_set['OnlineBackup'] = transforForamt(data_set['OnlineBackup'],dic_3)
data_set['DeviceProtection'] = transforForamt(data_set['DeviceProtection'],dic_3)
data_set['TechSupport'] = transforForamt(data_set['TechSupport'],dic_3)
data_set['StreamingTV'] = transforForamt(data_set['StreamingTV'],dic_3)
data_set['StreamingMovies'] = transforForamt(data_set['StreamingMovies'],dic_3)

dic_4={"No":0,"DSL":1, "Fiber optic":3}
data_set['InternetService'] = transforForamt(data_set['InternetService'],dic_4)


dic_5={"Month-to-month":1,"One year":2, "Two year":3}
data_set['Contract'] = transforForamt(data_set['Contract'],dic_5)

dic_6={"Electronic check":1,"Mailed check":2, "Bank transfer (automatic)":3, "Credit card (automatic)":4}
data_set['PaymentMethod'] = transforForamt(data_set['PaymentMethod'],dic_6)

dic_7={"No phone service":-1,"Yes":1, "No":0}
data_set['MultipleLines'] = transforForamt(data_set['MultipleLines'],dic_7)


# save the original dataset in one piece
data_set.to_csv("data/Telco_Churn.csv")


data_suffle = data_set.sample(frac=1)
train_ratio = 0.638
train_idx = int(train_ratio * data_suffle.shape[0])

train_data = data_suffle[0:train_idx]
test_data = data_suffle[train_idx+1:-1]
dataframe_train = pd.DataFrame(train_data)
dataframe_train.to_csv("data/train.csv")
dataframe_test = pd.DataFrame(test_data)
dataframe_test.to_csv("data/test.csv")




