import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data_train = pd.read_csv('Data_Train.csv',index_col='Unique_ID')
data_test = pd.read_csv('Data_Test.csv',index_col='Unique_ID')

### Dropping Country column
data_train_1 = data_train.drop('Country', axis=1)
data_test_1 = data_test.drop('Country', axis=1)

### Dropping Song_Name
data_train_2 = data_train_1.drop('Song_Name',axis=1)
data_test_2 = data_test_1.drop('Song_Name',axis=1)


### Converting Timestamp column in timestamp
data_train_2['Timestamp'] = pd.to_datetime(data_train_2['Timestamp'])
data_test_2['Timestamp'] = pd.to_datetime(data_test_2['Timestamp'])


data_train_2['year'] = pd.DatetimeIndex(data_train_2['Timestamp']).year
data_train_2['month'] = pd.DatetimeIndex(data_train_2['Timestamp']).month
data_train_2['dow'] = pd.DatetimeIndex(data_train_2['Timestamp']).dayofweek
data_train_2['day'] = pd.DatetimeIndex(data_train_2['Timestamp']).day


data_test_2['year'] = pd.DatetimeIndex(data_test_2['Timestamp']).year
data_test_2['month'] = pd.DatetimeIndex(data_test_2['Timestamp']).month
data_test_2['dow'] = pd.DatetimeIndex(data_test_2['Timestamp']).dayofweek
data_test_2['day'] = pd.DatetimeIndex(data_test_2['Timestamp']).day

data_train_3 = data_train_2.drop('Timestamp', axis=1)
data_test_3 = data_test_2.drop('Timestamp', axis=1)


### Dropping Name
data_train_4 = data_train_3.drop('Name', axis=1)
data_test_4 = data_test_3.drop('Name', axis=1)


### Handle Popularity column
def handleNumbers(txt):
    
    if(',' in txt):
        txt = txt.replace(",","")
    if('K' in txt):
        txt = txt.replace("K","")
        txt = float(txt) * 1000
    elif('M' in txt):
        txt = txt.replace("M","")
        txt = float(txt) * 100000
    return int(txt)


data_train_4['Popularity'] = data_train_4['Popularity'].apply(handleNumbers) 
data_test_4['Popularity'] = data_test_4['Popularity'].apply(handleNumbers)
data_train_4['Likes'] = data_train_4['Likes'].apply(handleNumbers) 
data_test_4['Likes'] = data_test_4['Likes'].apply(handleNumbers) 


data_train_5 = data_train_4.drop('Genre', axis=1)
data_test_5 = data_test_4.drop('Genre', axis=1)

y = data_train_5['Views']
X = data_train_5.drop('Views', axis = 1) 



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


from xgboost import XGBRegressor

def xgboost_model():
    regressor = XGBRegressor(objective='reg:squarederror', learning_rate = 0.1, n_estimators=500, seed=1729,n_jobs=8)
    return regressor

model = xgboost_model()

model.fit(X_train, y_train)

preds = model.predict(X_test)
model.score(X_test, y_test)


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, preds))

X_result = data_test_5.to_numpy()
y_output = model.predict(X_result)

data_test['Views'] = y_output

result = data_test[['Views']]
result.to_excel('output.xlsx')
