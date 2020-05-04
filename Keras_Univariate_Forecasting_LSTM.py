#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#### USER INPUT SECTION ###

# Global Variables #
global_source_data = "C:/Users/debas/Desktop/My Docs/Keras Code Development/RNN/Google_Stock_Price.csv"
global_ts_column = "Price"
global_date_column = "Date"
global_n_ahead = 30

param_look_back_period = 60
param_epoch = 500
param_drop_out = 0.15


# In[ ]:


### IMPORT ALL NECCESSARY PACKAGES ###

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


# In[ ]:


### DATA IMPORT ###

def data_import(source_data, date_col, hold_out):
    
    df = pd.read_csv(source_data,
                     header = 0,
                     index_col= date_col,
                     parse_dates=True,
                     infer_datetime_format=True,
                     squeeze=True)
    df = df.fillna(method='ffill')
    df = df.astype('float32').to_numpy()
    
    train_1darray = df[:len(df)-hold_out]
    test_1darray = df[len(df)-hold_out:len(df)]
 
    return train_1darray, test_1darray


# In[ ]:


### DATA PREPARATION ###

def data_prep(train, test, look_back): 
    
    # Standardizing Training Data In To A 2-D Array
    train_2d = train.reshape(-1, 1)
    scaler_obj = MinMaxScaler(feature_range=(0, 1))
    train_normal = scaler_obj.fit_transform(train_2d)
    
    # Creating Lag Variables In Training Data  
    train_x = []
    train_y = []
    for i in range(look_back, len(train_normal)):
        train_x.append(train_normal[(i-look_back):i, 0])
        train_y.append(train_normal[i, 0])
    train_x, train_y = np.array(train_x), np.array(train_y)
    
    # Converting In To 3-D Matrix
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    
    # Appending Training & Testing Datasets
    data = np.concatenate((train, test))
    data_2d = data.reshape(-1, 1)  
    
    # Standardizing Testing Data In To A 2-D Array
    data_normal = scaler_obj.transform(data_2d)
    
    # Creating Lag Variables In Testing Data 
    test_x = []
    for i in range(len(train), len(data_normal)):
        test_x.append(data_normal[(i-look_back):i, 0])
    test_x = np.array(test_x)
    
    # Converting In To 3-D Matrix
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    
    return train_x, train_y, test_x, scaler_obj


# In[ ]:


### MODEL DEVELOPMENT: LSTM ###

def model_lstm(train_x, train_y, test_x, test_y, look_back, epoch, drop_out, scaler):
    
    #Initialising the RNN
    regressor = Sequential()
    
    # Adding the input layer and the LSTM layer
    regressor.add(LSTM(units = int(look_back/2), 
                       return_sequences = True, 
                       input_shape = (look_back, 1),
                       kernel_initializer='uniform',
                       activation = 'relu', 
                       bias_initializer='zeros'))
    regressor.add(Dropout(rate = drop_out))
    
    # Adding a second LSTM layer
    regressor.add(LSTM(units = int(look_back/2), 
                       return_sequences = True, 
                       kernel_initializer='uniform',
                       activation = 'relu', 
                       bias_initializer='zeros'))
    regressor.add(Dropout(rate = drop_out))
    
    # Adding the output layer
    regressor.add(LSTM(units = 1, 
                       kernel_initializer='uniform',
                       activation = 'relu', 
                       bias_initializer='zeros'))
    
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    regressor.fit(train_x, train_y, epochs=epoch, batch_size=32)
    
    # Generating Predictions
    test_predictions = scaler.inverse_transform(regressor.predict(test_x))[:,0]
    mape = np.mean((abs(test_y-test_predictions)/test_y)*100)
    print("\nTest Sample MAPE: %.3f Percent\n" %mape)
    test_sample_output = pd.DataFrame({"Actuals" : test_y,"Predicted" : test_predictions})
    
    return test_sample_output


# In[ ]:


## Importing The Data 
train_1d, test_1d = data_import(global_source_data, 
                                global_date_column,
                                global_n_ahead)

## Data Preparation
model_train_x, model_train_y, model_val_x, scaler = data_prep(train_1d, 
                                                              test_1d, 
                                                              param_look_back_period)

## Running The Model
test_sample_output = model_lstm(model_train_x, 
                                model_train_y, 
                                model_val_x, 
                                test_1d, 
                                param_look_back_period, 
                                param_epoch, 
                                param_drop_out, scaler)

## Saving The Test Sample Output
test_sample_output.to_csv("Output.csv")


# In[ ]:




