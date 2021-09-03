#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from functools import reduce

from darts import TimeSeries
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    ExponentialSmoothing,
    ARIMA,
    AutoARIMA,
    RegressionEnsembleModel,
    RegressionModel,
    Theta,
    FFT,
    RNNModel
)

from darts.metrics import mape, mase, mae, rmse
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from darts.datasets import AirPassengersDataset

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import pickle


# # Delta 1

# In[2]:


name = 'Production'


# In[3]:


delta = 1


# In[4]:


df = pd.read_csv("monthly-milk-production.csv", parse_dates=["Month"], index_col="Month")


# In[5]:


df


# In[6]:


series = TimeSeries.from_series(df[name])


# ## Darts

# In[7]:


models = [ExponentialSmoothing(), AutoARIMA(), Prophet()]


# In[8]:


import functools

backtests = []

for model in models:
    print(f"{model}: running...")
    initial_time = time.time()
    hist_pred = model.historical_forecasts(series,
                            start=pd.Timestamp('1974-01-01'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
    hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
    backtests.append(hist_pred)
    
    final_time = time.time() - initial_time
    print(f"{model}: final time spent: {round(final_time, 3)}")


# In[9]:


from darts.dataprocessing.transformers import Scaler

print(f"LSTM: running...")
initial_time = time.time()
transformer = Scaler()
transformed_series = transformer.fit_transform(series)
lstm = RNNModel(model='LSTM', input_chunk_length=round(len(series)/4), output_chunk_length=1)
models.append(lstm)

hist_pred =  lstm.historical_forecasts(transformed_series,
                            start=pd.Timestamp('1974-01-01'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
backtests.append(transformer.inverse_transform(hist_pred))
final_time = time.time() - initial_time
print(f"LSTM: final time spent: {round(final_time, 3)}")


# In[10]:


darts_maes = {}
darts_rmses = {}

for i, m in enumerate(models):
    prediction = backtests[i]
    err_mae = mae(backtests[i], series)
    err_rmse = rmse(backtests[i], series)
    darts_maes[m] = err_mae
    darts_rmses[m] = err_rmse
    


# ## Timex

# In[11]:


with open(f"univariate/delta_{delta}/historical_predictions.pkl", 'rb') as file:
    p = pickle.load(file)


# In[12]:


timex_maes = {}
timex_rmses = {}

for i, m in enumerate(p):
    pred_timex = p[m]
    pred_timex = pred_timex[name].astype('float')
    pred_timex = TimeSeries.from_series(pred_timex)
    pred_timex = pred_timex.slice_intersect(backtests[0])
    assert len(pred_timex) == len(backtests[0])
    
    err_mae = mae(pred_timex, series)
    err_rmse = rmse(pred_timex, series)
    
    timex_maes[m] = err_mae
    timex_rmses[m] = err_rmse


# In[1]:


print("########## FINAL RESULTS ##########")
print(f"Case: Monthly Milk Production, delta: {delta}")
print("MAES")
print("Darts results:")
for i, m in enumerate(darts_maes):
    print(f"{m}, MAE={round(darts_maes[m], 3)}")

print("Timex results:")
for i, m in enumerate(timex_maes):
    print(f"{m}, MAE={round(timex_maes[m], 3)}")

print("------------------------")
print("RMSES")
print("Darts results:")

for i, m in enumerate(darts_rmses):
    print(f"{m}, RMSE={round(darts_rmses[m], 3)}")

print("Timex results:")
for i, m in enumerate(timex_rmses):
    print(f"{m}, RMSE={round(timex_rmses[m], 3)}")


# # Delta 3

# In[2]:


name = 'Production'


# In[3]:


delta = 3


# In[4]:


df = pd.read_csv("monthly-milk-production.csv", parse_dates=["Month"], index_col="Month")


# In[5]:


df


# In[6]:


series = TimeSeries.from_series(df[name])


# ## Darts

# In[7]:


models = [ExponentialSmoothing(), AutoARIMA(), Prophet()]


# In[8]:


import functools

backtests = []

for model in models:
    print(f"{model}: running...")
    initial_time = time.time()
    hist_pred = model.historical_forecasts(series,
                            start=pd.Timestamp('1974-01-01'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
    hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
    backtests.append(hist_pred)
    
    final_time = time.time() - initial_time
    print(f"{model}: final time spent: {round(final_time, 3)}")


# In[9]:


from darts.dataprocessing.transformers import Scaler

print(f"LSTM: running...")
initial_time = time.time()
transformer = Scaler()
transformed_series = transformer.fit_transform(series)
lstm = RNNModel(model='LSTM', input_chunk_length=round(len(series)/4), output_chunk_length=1)
models.append(lstm)

hist_pred =  lstm.historical_forecasts(transformed_series,
                            start=pd.Timestamp('1974-01-01'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
backtests.append(transformer.inverse_transform(hist_pred))
final_time = time.time() - initial_time
print(f"LSTM: final time spent: {round(final_time, 3)}")


# In[10]:


darts_maes = {}
darts_rmses = {}

for i, m in enumerate(models):
    prediction = backtests[i]
    err_mae = mae(backtests[i], series)
    err_rmse = rmse(backtests[i], series)
    darts_maes[m] = err_mae
    darts_rmses[m] = err_rmse
    


# ## Timex

# In[11]:


with open(f"univariate/delta_{delta}/historical_predictions.pkl", 'rb') as file:
    p = pickle.load(file)


# In[12]:


timex_maes = {}
timex_rmses = {}

for i, m in enumerate(p):
    pred_timex = p[m]
    pred_timex = pred_timex[name].astype('float')
    pred_timex = TimeSeries.from_series(pred_timex)
    pred_timex = pred_timex.slice_intersect(backtests[0])
    assert len(pred_timex) == len(backtests[0])
    
    err_mae = mae(pred_timex, series)
    err_rmse = rmse(pred_timex, series)
    
    timex_maes[m] = err_mae
    timex_rmses[m] = err_rmse


# In[1]:


print("########## FINAL RESULTS ##########")
print(f"Case: Monthly Milk Production, delta: {delta}")
print("MAES")
print("Darts results:")
for i, m in enumerate(darts_maes):
    print(f"{m}, MAE={round(darts_maes[m], 3)}")

print("Timex results:")
for i, m in enumerate(timex_maes):
    print(f"{m}, MAE={round(timex_maes[m], 3)}")

print("------------------------")
print("RMSES")
print("Darts results:")

for i, m in enumerate(darts_rmses):
    print(f"{m}, RMSE={round(darts_rmses[m], 3)}")

print("Timex results:")
for i, m in enumerate(timex_rmses):
    print(f"{m}, RMSE={round(timex_rmses[m], 3)}")


# # Delta 6

# In[2]:


name = 'Production'


# In[3]:


delta = 6


# In[4]:


df = pd.read_csv("monthly-milk-production.csv", parse_dates=["Month"], index_col="Month")


# In[5]:


df


# In[6]:


series = TimeSeries.from_series(df[name])


# ## Darts

# In[7]:


models = [ExponentialSmoothing(), AutoARIMA(), Prophet()]


# In[8]:


import functools

backtests = []

for model in models:
    print(f"{model}: running...")
    initial_time = time.time()
    hist_pred = model.historical_forecasts(series,
                            start=pd.Timestamp('1974-01-01'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
    hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
    backtests.append(hist_pred)
    
    final_time = time.time() - initial_time
    print(f"{model}: final time spent: {round(final_time, 3)}")


# In[9]:


from darts.dataprocessing.transformers import Scaler

print(f"LSTM: running...")
initial_time = time.time()
transformer = Scaler()
transformed_series = transformer.fit_transform(series)
lstm = RNNModel(model='LSTM', input_chunk_length=round(len(series)/4), output_chunk_length=1)
models.append(lstm)

hist_pred =  lstm.historical_forecasts(transformed_series,
                            start=pd.Timestamp('1974-01-01'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
backtests.append(transformer.inverse_transform(hist_pred))
final_time = time.time() - initial_time
print(f"LSTM: final time spent: {round(final_time, 3)}")


# In[10]:


darts_maes = {}
darts_rmses = {}

for i, m in enumerate(models):
    prediction = backtests[i]
    err_mae = mae(backtests[i], series)
    err_rmse = rmse(backtests[i], series)
    darts_maes[m] = err_mae
    darts_rmses[m] = err_rmse
    


# ## Timex

# In[11]:


with open(f"univariate/delta_{delta}/historical_predictions.pkl", 'rb') as file:
    p = pickle.load(file)


# In[12]:


timex_maes = {}
timex_rmses = {}

for i, m in enumerate(p):
    pred_timex = p[m]
    pred_timex = pred_timex[name].astype('float')
    pred_timex = TimeSeries.from_series(pred_timex)
    pred_timex = pred_timex.slice_intersect(backtests[0])
    assert len(pred_timex) == len(backtests[0])
    
    err_mae = mae(pred_timex, series)
    err_rmse = rmse(pred_timex, series)
    
    timex_maes[m] = err_mae
    timex_rmses[m] = err_rmse


# In[1]:


print("########## FINAL RESULTS ##########")
print(f"Case: Monthly Milk Production, delta: {delta}")
print("MAES")
print("Darts results:")
for i, m in enumerate(darts_maes):
    print(f"{m}, MAE={round(darts_maes[m], 3)}")

print("Timex results:")
for i, m in enumerate(timex_maes):
    print(f"{m}, MAE={round(timex_maes[m], 3)}")

print("------------------------")
print("RMSES")
print("Darts results:")

for i, m in enumerate(darts_rmses):
    print(f"{m}, RMSE={round(darts_rmses[m], 3)}")

print("Timex results:")
for i, m in enumerate(timex_rmses):
    print(f"{m}, RMSE={round(timex_rmses[m], 3)}")


# In[37]: