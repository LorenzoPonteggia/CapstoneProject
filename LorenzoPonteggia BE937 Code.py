import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams['figure.figsize'] = [8, 4]
plt.rcParams['figure.dpi'] = 300

df = pd.read_csv('/modeldata.csv')
date = df['Date']
date = date.drop([0,1,2])
date = pd.to_datetime(date, format="%d.%m.%Y")

cols = list(df)[0:106]
df = df[cols].astype(float)

#plotting the transfromed Y to confirm it is stationary
plt.title('CPIAUCSL')
plt.plot(date, df['CPIAUCSL'][3:])
plt.xlabel('Date', fontsize=8)
plt.ylabel('Index Value', fontsize=8)
plt.tight_layout()
plt.show()

# loop through each column of the DataFrame
for col in cols:
    if df.loc[0, col] == 1:
        # get level
        df[col] = df[col].diff()
    
    elif df.loc[0, col] == 2:
        # get change
        df[col] = df[col].pct_change()
        
    elif df.loc[0, col] == 3:
        # get logarithm
        df[col] = np.log(df[col])
        
    elif df.loc[0, col] == 5:
        # get first difference of logarithm
        df[col] = np.log(df[col]).diff()
        
    elif df.loc[0, col] == 6:
        # get second difference of logarithm
        df[col] = (np.log(df[col]).diff()).diff()

df = pd.concat([df, date], axis=1)
df = df.drop([0,1,2,3])
date = date.drop([3])

x = df.iloc[:,:-2]
df['CPIAUCSL'][3]=0
y = df['CPIAUCSL']

#plotting the transfromed Y to confirm it is stationary
plt.title('CPIAUCSL After Transformation')
plt.plot(date, y)
plt.xlabel('Date', fontsize=8)
plt.ylabel('Index rate of change', fontsize=8)
plt.tight_layout()
plt.show()

#splitting dataset------------------------------------------------------------------------------------
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.80, shuffle=False)

#variables to accomodate changes in train/test size---------------------------------------------
trainL = len(train_y)
L = len(test_y)

#LSTM------------------------------------------------------------------------------------------
# normalize the dataset
df = df.iloc[:,:-1]
scaler = StandardScaler()
scaler = scaler.fit(df)
df_for_training_scaled = scaler.transform(df)

#Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1   # Number of months we want to forecast
n_past = trainL  # Number of past months we want to use to forecast

#Reformat input data into a shape: (n_samples x timesteps x n_features)
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# define the Autoencoder model

model = Sequential()
model.add(LSTM(trainL, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(8, activation='relu', return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()


# fit the model
History = model.fit(trainX, trainY, epochs=8, batch_size=16, validation_split=0.5, verbose=1)

n_past = 28
n_days_for_prediction=L  #let us predict past 15 days

#Make prediction
prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction


#Perform inverse transformation to rescale back to original range
lstm_pred = np.repeat(prediction, df.shape[1], axis=-1)
lstm_pred = scaler.inverse_transform(lstm_pred)[:,0]

lstm_mse = mean_squared_error(test_y, lstm_pred)

#EXP stands for expanding window AR(1)-----------------------------------------------------------------------------------------------
forecast_horizon = 1

exp_predictions = np.zeros(L)
for t in range(L):
    train_data = y[:trainL+t]
    exp_model = sm.tsa.AutoReg(train_data, 1)
    results = exp_model.fit()
    forecast = results.forecast(steps=forecast_horizon)
    exp_predictions[t-forecast_horizon:t] = forecast
    
exp_resid = np.subtract(test_y, exp_predictions)
exp_mse = mean_squared_error(test_y, exp_predictions)
#slide stands for expanding window AR(1)-----------------------------------------------------------------------------------------------
# Perform sliding window forecast
window_size = 18
forecast_horizon = 1
slide_predictions = np.zeros(L)

for t in range(trainL, trainL+L):
    train_data = y[t-window_size:t]
    sliding_model = sm.tsa.AutoReg(train_data, 1)
    results = sliding_model.fit()
    forecast = results.forecast(steps=forecast_horizon)
    slide_predictions[t-trainL] = forecast


slide_resid = np.subtract(test_y, slide_predictions)

slide_mse = mean_squared_error(test_y, slide_predictions)

#Calculating MSE and RMSE-----------------------------------------------------------------------------------------------
print('LSTM MSE=')
print(lstm_mse)

print('slide AR(1) MSE=')
print(slide_mse)

print('AR(1) Expanding MSE=')
print(exp_mse)

#Ridge model-----------------------------------------------------------------------------------------------
ridge_model = linear_model.RidgeCV(cv=3).fit(train_x, train_y)

print('Ridge model Score=')
print(ridge_model.score(test_x,test_y))

ridge_pred = ridge_model.predict(test_x)

ridge_mse = mean_squared_error(test_y, ridge_pred)
print('Ridge MSE=')
print(ridge_mse)

#LASSO model-----------------------------------------------------------------------------------------------
lasso_model = linear_model.LassoCV(cv=2).fit(train_x, train_y)
lasso_pred = lasso_model.predict(test_x)



print('LASSO model Score=')
print(lasso_model.score(test_x,test_y))
lasso_pred = lasso_model.predict(test_x)

lasso_mse = mean_squared_error(test_y, lasso_pred) 
print('LASSO MSE=')
print(lasso_mse)

# #ENET model-----------------------------------------------------------------------------------------------
enet_model = linear_model.ElasticNetCV(cv=10).fit(train_x, train_y)
print('ENET model Score=')
print(enet_model.score(test_x,test_y))

enet_pred = enet_model.predict(test_x)

enet_mse = mean_squared_error(test_y, enet_pred) 
print('ENET MSE=')
print(enet_mse)

#calculating RMSE-----------------------------------------------------------------------------------------------
ridge_rmse = ridge_mse/slide_mse
print('Ridge RMSE=')
print(ridge_rmse)

lasso_rmse = lasso_mse/slide_mse
print('LASSO RMSE=')
print(lasso_rmse)

enet_rmse = enet_mse/slide_mse
print('ENET RMSE=')
print(enet_rmse)

lstm_rmse = lstm_mse/slide_mse
print('LSTM RMSE=')
print(lstm_rmse)

#-----------------------------------------------------------------------------------------------

date = date.values
results = pd.DataFrame({'Date': date[-L:], 'Actual': y[-L:],'AR(1) Sliding Window': slide_predictions, 'AR(1) Expanding Window': exp_predictions, 'Ridge': ridge_pred, 'LASSO': lasso_pred, 'ENET': enet_pred, 'LSTM': lstm_pred})
results = results.set_index(results.columns[0])

#-----------------------------------------------------------------------------------------------

#plotting AR(1) residuals
plt.title('AR(1) models residuals')
plt.plot(slide_resid, c='tab:orange')
plt.plot(exp_resid, c='tab:green')
plt.title('Residuals from AR(1) Models', fontsize=20)
plt.xlabel('Date', fontsize=8)
plt.ylabel('Inflation Rate', fontsize=8)
plt.legend(('Sliding Window AR(1)', 'Expanding Window AR(1)'), loc ="lower left", fontsize=8)
plt.show()

#plotting slide ar
plt.title('AR(1) with Sliding Window Predictions')
plt.rc('font', size=7)
plt.plot(results.iloc[:,0], c='tab:blue')
plt.plot(results.iloc[:,1], c='tab:orange')
plt.xlabel('Date', fontsize=8)
plt.ylabel('Inflation Rate', fontsize=8)
plt.legend(('Actual', 'AR(1) Sliding'), loc ="lower left", fontsize=8)
plt.show()

#plotting exp ar
plt.title('AR(1) with Expanding Window Predictions ')
plt.rc('font', size=7)
plt.plot(results.iloc[:,0], c='tab:blue')
plt.plot(results.iloc[:,2], c='tab:green')
plt.xlabel('Date', fontsize=8)
plt.ylabel('Inflation Rate', fontsize=8)
plt.legend(('Actual', 'AR(1) Expanding'), loc ="lower left", fontsize=8)
plt.show()

#plotting ars
plt.title('AR(1) Models')
plt.rc('font', size=7)
plt.plot(results.iloc[:,0], c='tab:blue')
plt.plot(results.iloc[:,1], c='tab:orange')
plt.plot(results.iloc[:,2], c='tab:green')
plt.xlabel('Date', fontsize=8)
plt.ylabel('Inflation Rate', fontsize=8)
plt.legend(('Actual', 'AR(1) Sliding', 'AR(1) Expanding'), loc ="lower left", fontsize=8)
plt.show()

#plotting ridge
plt.title('Ridge Predictions')
plt.rc('font', size=7)
plt.plot(results.iloc[:,0], c='tab:blue')
plt.plot(results.iloc[:,3], c='tab:red')
plt.xlabel('Date', fontsize=8)
plt.ylabel('Inflation Rate', fontsize=8)
plt.legend(('Actual', 'Ridge'), loc ="lower left", fontsize=8)
plt.show()

#plotting lasso
plt.title('LASSO Predictions')
plt.rc('font', size=7)
plt.plot(results.iloc[:,0], c='tab:blue')
plt.plot(results.iloc[:,4], c='tab:purple')
plt.xlabel('Date', fontsize=8)
plt.ylabel('Inflation Rate', fontsize=8)
plt.legend(('Actual', 'LASSO'), loc ="lower left", fontsize=8)
plt.show()
        
#plotting enet
plt.title('Elastic Net Predictions')
plt.rc('font', size=7)
plt.plot(results.iloc[:,0], c='tab:blue')
plt.plot(results.iloc[:,5], c='tab:brown')
plt.xlabel('Date', fontsize=8)
plt.ylabel('Inflation Rate', fontsize=8)
plt.legend(('Actual', 'Elastic Net'), loc ="lower left", fontsize=8)
plt.show()

#plotting ridge and lasso and enet
plt.title('Ridge, LASSO, and Elastic Net Predictions')
plt.rc('font', size=7)
plt.plot(results.iloc[:,0], c='tab:blue')
plt.plot(results.iloc[:,3], c='tab:red')
plt.plot(results.iloc[:,4], c='tab:purple')
plt.plot(results.iloc[:,5], c='tab:brown')
plt.xlabel('Date', fontsize=8)
plt.ylabel('Inflation Rate', fontsize=8)
plt.legend(('Actual', 'Ridge', 'LASSO', 'Elastic Net'), loc ="lower left", fontsize=8)
plt.show()

#plotting lstm
plt.title('LSTM Predictions')
plt.rc('font', size=7)
plt.plot(results.iloc[:,0], c='tab:blue')
plt.plot(results.iloc[:,6], c='tab:pink')
plt.xlabel('Date', fontsize=8)
plt.ylabel('Inflation Rate', fontsize=8)
plt.legend(('Actual', 'LSTM'), loc ="lower left", fontsize=8)
plt.show()


#plotting results
plt.title('Predictions')
plt.rc('font', size=7)
plt.plot(results)
plt.xlabel('Date', fontsize=8)
plt.ylabel('Inflation Rate', fontsize=8)
plt.legend(('Actual', 'AR(1) Sliding', 'AR(1) Expanding', 'Ridge', 'LASSO','Elastic Net', 'LSTM'), loc ="lower left", fontsize=8)
plt.show()

cridge=ridge_model.coef_
classo=lasso_model.coef_
cenet=enet_model.coef_

ridgeper=np.logical_or(np.logical_and(test_y > 0, ridge_pred > 0), np.logical_and(test_y < 0, ridge_pred < 0))
ridgecorr=0

for x in ridgeper:
    if x is True:
        ridgecorr= ridgecorr+1
        
lassoper=np.logical_or(np.logical_and(test_y > 0, lasso_pred > 0), np.logical_and(test_y < 0, lasso_pred < 0))
lassocorr=0

for x in lassoper:
    if x is True:
        lassocorr= lassocorr+1

enetper=np.logical_or(np.logical_and(test_y > 0, enet_pred > 0), np.logical_and(test_y < 0, enet_pred < 0))
enetcorr=0

for x in enetper:
    if x is True:
        enetcorr= enetcorr+1
    
slideper=np.logical_or(np.logical_and(test_y > 0, slide_predictions > 0), np.logical_and(test_y < 0, slide_predictions < 0))
slidecorr=0

for x in slideper:
    if x is True:
        slidecorr= slidecorr+1

expper=np.logical_or(np.logical_and(test_y > 0, exp_predictions > 0), np.logical_and(test_y < 0, exp_predictions < 0))
expcorr=0

for x in expper:
    if x is True:
        expcorr= expcorr+1

lstmper=np.logical_or(np.logical_and(test_y > 0, lstm_pred > 0), np.logical_and(test_y < 0, lstm_pred < 0))
lstmcorr=0

for x in lstmper:
    if x is True:
        lstmcorr= lstmcorr+1