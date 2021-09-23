# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:55:37 2021

Description:Implementing a SARIMA model to forecast the adjusted closing stock 
            prices of many companies using the yfinance library which gives 
            access to the data of companies stored in the yahoo finance page.

@author: Daniel Dorantes Sánchez
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

def MSE(forecast, real):
    """Function that shows the MSE of the forecast vs the real values
          Arguments:
            forecast - predicted values
            real - real values
          Return:
            means_error - value of the mean square error
    """
    forecast=np.array(forecast)
    real=np.array(real)
    forecast=np.exp(forecast)
    real=np.exp(real)
    error_acum=0
    for i in range(len(real)):
        error= real[i]-forecast[i]
        error_acum=error_acum+error**2
    means_error = error_acum/len(real)
    return means_error

def optSARIMA(params, s, train, test):
    """Function of optimization to obtain the dataframe with the best 
       parameters p,q,d and P,Q,D with the best AIC and SSE
       
       Arguments:
        params_a - list with (p, d, q, P, D, Q)
            p - autoregression order
            d - integration order
            q - moving average order 
            P - seasonal autoregression order
            D - seasonal integration order
            Q - seasonal moving average order
            s - length of season
        train - contains the dataset
        test - test data (real value to compare it)
        
      Return:
          order - List with the optimal values of params(p,d,q,P,D,Q)
    """
    res=[]
    for i in range(len(params)):
        par=params[i]
        try:
            model = SARIMAX(train, order=(par[0], par[1], par[2]), seasonal_order=(par[3], par[4], par[5], s)).fit(maxiter=50)
        except:
            continue
        #aic=model.aic
        st=len(test)
        e=len(train)+len(test)
        fcast = model.predict(start=st, end=e, dynamic=True)
        mse = MSE(fcast,test)
        res.append([par, mse])
    res_df = pd.DataFrame(res)
    res_df.columns = ['(p,d,q)x(P,D,Q)', 'MSE']
    #Sort in ascending order, lower MSE is better
    res_df = res_df.sort_values(by='MSE', ascending=True).reset_index(drop=True)
    print("List of all resulting MSE for each iteration of params")
    print(res_df)
    order= res_df.iloc[0][0]
    return order


#Download the data using yfinance library
df= yf.download('MSFT',start='2016-01-01',end='2021-01-01',interval="1mo")
df.dropna(inplace=True)

#Get only the Adjusted Close price column
data= pd.DataFrame(df,columns = ['Adj Close'])
print(data)

#Display the data in a plot
plt.figure(figsize=(16,8))
plt.title('Adjusted Close stock price of MSFT')
plt.plot(data)
plt.show()

#Calculate the natural logarithm of the data so that the first difference 
#gets the monthly data percentage growth.
data = np.log(data)
#print(data)

#Split the data into train and test 
#Get the last 12 months for testing and the rest for training
data_train=data[:-12]
data_test=data[-12:]
#print(data_train)
#print(data_test)

#Define parameters p,d,q and P,D,Q assume season length is 12
p = range(0, 3, 1)
d = range(0, 2, 1)
q = range(0, 3, 1)
P = range(0, 2, 1)
D = range(1, 2, 1) #Force to make a seasonal difference
Q = range(0, 2, 1)
s = 12
#Put together the parameters into a list
params = product(p, d, q, P, D, Q)
params_list = list(params)
#print(len(params_list))

#Call the optSARIMA func. to get the optimal parameters p, d, q, P, D, Q
order = optSARIMA(params_list, s, data_train,data_test)
or_a = order[:3]
or_s = order[3:]
#print(order)
#print(or_a)
#print(or_s)

#Build Model to test the result with the data_test
model_fit = SARIMAX(data_train, order=or_a, seasonal_order=(or_s[0],or_s[1],or_s[2], s)).fit()

#Test the model with predict function
st=len(data_train)
e=len(data)
forecast = model_fit.predict(start=st, end=e, dynamic=True)
#print(data_test)
#print(forecast)
#Get the Mean Square Error
mean_serror= MSE(forecast,data_test)
print("Mean Square Error",mean_serror)

#Put all data to its original values
data_train= np.exp(data_train)
data_test= np.exp(data_test)
forecast= np.exp(forecast)

#Plot forecast to compare the prediction with the test data (real values)
plt.plot(data_train, label='training')
plt.plot(data_test, label='actual')
plt.plot(forecast, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper ledt', fontsize=8)
plt.show()

#Make madel to predict sales in the next 12 months
prediction = SARIMAX(data, order=or_a, seasonal_order=(or_s[0],or_s[1],or_s[2], s)).fit()

#Make prediction for the future 12 months
st=len(data)
e=len(data)+12
prediction = prediction.predict(start=st, end=e, dynamic=True)
data=np.exp(data)
prediction= np.exp(prediction)

#Plot prediction 12 months ahead
plt.plot(data, label='data')
plt.plot(prediction, label='prediction')
plt.title('Forecast of the next 12 months')
plt.legend(loc='upper ledt', fontsize=8)
plt.show()  