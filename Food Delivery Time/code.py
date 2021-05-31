import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
import seaborn as sns
sns.set()
from sklearn import preprocessing


train_data = pd.read_excel('Data_Train.xlsx',index_col ='Restaurant')
test_data = pd.read_excel('Data_Test.xlsx',index_col ='Restaurant')

data = pd.concat([train_data, test_data], sort=False)


distinc_cuisin = []

for index, row in data.iterrows(): 
    cusines = row['Cuisines'].split(",")
    #print (data.columns) 
    #print (index)
    for i in cusines: 
        c = i.replace(" ", "").lower()
        #distinc_cuisin.append(c)
        data.at[index,c] = 1
        if c not in distinc_cuisin:
            distinc_cuisin.append(c)
			

for col in distinc_cuisin:
    data[col] = data[col].fillna(0)
	
	
data_1 = data.drop('Cuisines',axis=1)

data_1['Average_Cost'] = data_1.Average_Cost.replace({r'(.*?)([0-9]+)' : r'\2'}, regex=True)

data_1['Average_Cost'] =  pd.to_numeric(data_1['Average_Cost'], errors ='coerce')

data_1['Average_Cost'] = data_1['Average_Cost'].fillna(data_1['Average_Cost'].mean())


data_1['Minimum_Order'] = data_1.Minimum_Order.replace({r'(.*?)([0-9]+)' : r'\2'}, regex=True)

data_1['Minimum_Order'] =  pd.to_numeric(data_1['Minimum_Order'], errors ='coerce')
data_1['Minimum_Order'] = data_1['Minimum_Order'].fillna(data_1['Minimum_Order'].mean())


data_1['Rating'] = data_1.Rating.replace({r'(.*?)([0-9]+)' : r'\2'}, regex=True)

data_1['Rating'] =  pd.to_numeric(data_1['Rating'], errors ='coerce')
data_1['Rating'] = data_1['Rating'].fillna(data_1['Rating'].mean())



data_1 = data_1.drop('Votes', axis=1)


data_1 = data_1.drop('Reviews', axis=1)


data_1['Delivery_Time'] = data_1['Delivery_Time'].map({'10 minutes':0,'20 minutes': 1, '30 minutes' :2,  '45 minutes' :3, '65 minutes':4, '80 minutes' :5, '120 minutes' :6})


data_1 = pd.get_dummies(data_1, drop_first=True)

sns.distplot(data_1['Average_Cost'])


y_dataframe = data_1[['Delivery_Time']]
x_dataframe = data_1.drop('Delivery_Time',axis=1)

scaled_df_inputs = preprocessing.scale(x_dataframe)

X_train = scaled_df_inputs[:11094, :]
X_test = scaled_df_inputs[11094:, :]
Y_train = y_dataframe.iloc[:11094, :]


inputs = X_train
targets = Y_train.to_numpy()



samples_count = inputs.shape[0]

train_samples_count = int(0.80 * samples_count)
validation_samples_count = int(0.10 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = inputs[:train_samples_count]
train_targets = targets[:train_samples_count]

validation_inputs = inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = inputs[train_samples_count+validation_samples_count:]
test_targets = targets[train_samples_count+validation_samples_count:]



from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size = 0.25, random_state = 21)
from sklearn.ensemble import RandomForestClassifier
clasifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state = 0)

regressor.fit(X_train, y_train)
test_output = regressor.predict(X_test)
validation_output = regressor.predict(validation_inputs)
