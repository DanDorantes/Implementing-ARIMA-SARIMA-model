# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 22:47:18 2021

Description: Implementing a SARIMA model to predict sales of products
             in the category of candy.
Note:The data used is a sample dataset of main consumer products from main 
competitors of the category of candy. This dataset has historical sales of 
the main 7 products. The dataset was created using real data but changing 
the name of products for confidentiality. For each product you have sales in 
value ($) and sales in volume (kgs).

@author: Daniel Dorantes Sánchez
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import math

def MSE(forecast, real):
    """Function that shows the MSE of the forecast
          Arguments:
            forecast - numpy array with the predicted values
            real - numpy array with the real values
          Return:
            means_error - value of the mean square error
    """
    forecast=np.array(forecast)
    real=np.array(real)
    #Get the exponential of the real and forecast to get the original values
    forecast=np.exp(forecast)
    real=np.exp(real)
    error_acum=0
    for i in range(len(real)):
        error= real[i]-forecast[i]
        error_acum=+error**2
    means_error = error_acum/len(real)
    return means_error

def optSARIMA(params, s,train,test):
    """Function of optimization to obtain the best parameters p,q,d and P,Q,D 
       by choosing the parameters that gives the smallest MSE
       
       Arguments:
        parameters_list - list with (p, d, q, P, D, Q)
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
            model = SARIMAX(train, order=(par[0], par[1], par[2]), seasonal_order=(par[3], par[4], par[5], s)).fit()
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
    
    return res_df

#Read the data frame
df= pd.read_excel(r'salesfabs.xlsx')
dataset=df['vtotal']
#print(dataset)

plt.figure(figsize=(16,8))
plt.title('Total Sales')
plt.plot(df['vtotal'])
plt.show()

#Calculate the natural logarithm of the data so that the first difference 
#gets the monthly data percentage growth.
data = np.log(dataset)
#print(data)

#Split the data into train and test 75:25
t=math.floor(len(data)*0.25)
data_train=data[:-t]
data_test=data[-t:]
#print(data_train)
#print(data_test)

#Define range of parameters p,d,1 and P,D,Q assume season length is 12
p = range(0, 3, 1)
d = range(0, 2, 1)
q = range(0, 3, 1)
P = range(0, 2, 1)
D = range(0, 2, 1)
Q = range(0, 2, 1)
s = 12
#Put together the parameters into a list
params = product(p, d, q, P, D, Q)
params_list = list(params)
#print(len(params_list))

#Call the optSARIMA func. to get the optimal parameters p, d, q, P, D, Q
res_df = optSARIMA(params_list, s, data_train,data_test)
print(res_df)
order= res_df.iloc[0][0]
or_a = order[:3]
or_s = order[3:]
print(or_a)
print(or_s)

#Build Model to test the result
model_fit = SARIMAX(data_train, order=or_a, seasonal_order=(or_s[0],or_s[1],or_s[2], s)).fit()

#Test the model using predict function
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

#Plot forecast to predict the test data
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
plt.title('Prediction of the next 12 months')
plt.legend(loc='upper ledt', fontsize=8)
plt.show()

