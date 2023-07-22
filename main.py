import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
list_of_names = ['file_fyp_t', 'usama']
from xgboost import XGBClassifier
# Creating Empty list
df_list = []

# Appending datasets into the list
for i in range(len(list_of_names)):
    temp_df = pd.read_csv("E:" + list_of_names[i] + ".csv")
    df_list.append(temp_df)
df=df_list[0]
df['YEAR']=df.YEAR.astype(str)
df['MO']=df.MO.astype(str)
df['DY']=df.DY.astype(str)

df['date']=df['YEAR'].str.cat(df['MO'], sep='/')
df['Date']=df['DY'].str.cat(df['MO'], sep='/')
df['Date']=df['Date'].str.cat(df['YEAR'],sep='/')
df.set_index(['Date'],inplace= True)
df.drop(columns=['YEAR','MO','DY','Condition'],axis=1,inplace=True)
df.drop(columns=['date'],axis=1,inplace=True)

scaler=MinMaxScaler()
df_scaled = scaler.fit_transform(df)
print('Scaled df:\n',df_scaled,'\n',df_scaled.shape)

train,test = train_test_split(df_scaled,test_size=0.4, shuffle=False)
x_train,y_train, x_test, y_test = [],[],[],[]
for i in range(1,len(train)):
    x_train.append(train[i-1])
    y_train.append(train[i])
for i in range(1,len(test)):
    x_test.append(train[i-1])
    y_test.append(train[i])

x_train, y_train = np.array(x_train),np.array(y_train)
x_test, y_test = np.array(x_test),np.array(y_test)

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import Dropout

model=Sequential()
model.add(Dense(4, input_dim =4, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(4))
print(model.summary())
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=200,batch_size=15,shuffle = False)
model.save('my_model.h5')

data1=df_list[1]
import warnings
warnings.filterwarnings('ignore')
countsun=len(data1[data1.Condition=='sun'])
countcloud=len(data1[data1.Condition=='cloud'])
countrain=len(data1[data1.Condition=='rain'])
countdrizzle=len(data1[data1.Condition=='drizzle'])
data1=data1.drop(['YEAR','MO','DY','Date'],axis=1)
#sns.heatmap(data1.corr(),annot=True,cmap='coolwarm')
lc=LabelEncoder()
print(data1['Condition'])
data1['Condition']=lc.fit_transform(data1['Condition'])
data1.drop(data1.columns[data1.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
x = ((data1.loc[:, data1.columns != 'Condition']).astype(int)).values[:, 0:]
y = data1['Condition'].values
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
xgbc=XGBClassifier(n_estimators=120,subsample=0.5,colsample_bytree=0.5,gamma=0.3,learning_rate=0.1,max_depth=3,min_child_weight=1)
model=xgbc.fit(x_test,y_test)

file_name = "model.pickle"

# Dump the model object into a file using pickle.dump() in a single line
pickle.dump(model, open(file_name, "wb"))

print(f"Model dumped into file '{file_name}' successfully!")
