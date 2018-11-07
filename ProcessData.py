import pandas as pd

def transforForamt(dataset, diction):
    for i in range(len(dataset)):
        dataset[i] = diction[dataset[i]]
    return dataset


data_set = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
dic={"Female":1,"Male":0}
data_set['gender'] = transforForamt(data_set['gender'],dic)
data_suffle = data_set.sample(frac=1)
train_ratio = 0.638
train_idx = int(train_ratio * data_suffle.shape[0])

train_data = data_suffle[0:train_idx]
test_data = data_suffle[train_idx+1:-1]
dataframe_train = pd.DataFrame(train_data)
dataframe_train.to_csv("data/train.csv")
dataframe_test = pd.DataFrame(train_data)
dataframe_test.to_csv("data/test.csv")

data_train  = pd.read_csv('data/train.csv')

print(data_set.shape)




