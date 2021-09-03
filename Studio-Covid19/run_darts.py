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


# # Daily cases

# ## Delta 1

# In[48]:


name = 'Daily cases'


# In[49]:


delta = 1


# In[50]:


df = pd.read_csv("Covid19-italy.csv", parse_dates=["Date"], index_col="Date")


# In[51]:


df


# In[52]:


df = df.loc[:pd.Timestamp("20210225"), :]


# In[53]:


df


# In[54]:


series = TimeSeries.from_series(df[name])


# ## Darts

# In[10]:


models = [ExponentialSmoothing(), AutoARIMA(), Prophet()]


# In[ ]:


import functools

backtests = []

for model in models:
    print(f"{model}: running...")
    initial_time = time.time()
    hist_pred = model.historical_forecasts(series,
                            start=pd.Timestamp('2020-08-19'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
    hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
    backtests.append(hist_pred)
    
    final_time = time.time() - initial_time
    print(f"{model}: final time spent: {round(final_time, 3)}")


# In[ ]:


from darts.dataprocessing.transformers import Scaler

print(f"LSTM: running...")
initial_time = time.time()
transformer = Scaler()
transformed_series = transformer.fit_transform(series)
lstm = RNNModel(model='LSTM', input_chunk_length=round(len(series)/4), output_chunk_length=1)
models.append(lstm)

hist_pred =  lstm.historical_forecasts(transformed_series,
                            start=pd.Timestamp('2020-08-19'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
backtests.append(transformer.inverse_transform(hist_pred))
final_time = time.time() - initial_time
print(f"LSTM: final time spent: {round(final_time, 3)}")


# In[ ]:


darts_maes = {}
darts_rmses = {}

for i, m in enumerate(models):
    prediction = backtests[i]
    #     print(prediction)
    err_mae = mae(backtests[i], series)
    err_rmse = rmse(backtests[i], series)
    darts_maes[m] = err_mae
    darts_rmses[m] = err_rmse


# ## Timex

# In[ ]:


with open(f"univariate/delta_{delta}/historical_predictions.pkl", 'rb') as file:
    p = pickle.load(file)


# In[ ]:


timex_maes = {}
timex_rmses = {}

for i, m in enumerate(p):
    pred_timex = p[m]
    pred_timex = pred_timex[name].astype('float')
    pred_timex = TimeSeries.from_series(pred_timex)
    pred_timex = pred_timex.slice_intersect(backtests[0])
    #     pred_timex = pred_timex.drop_after(backtests[0].time_index()[-1] + pd.Timedelta(days=1))
    assert len(pred_timex) == len(backtests[i])

    err_mae = mae(pred_timex, series)
    err_rmse = rmse(pred_timex, series)

    timex_maes[m] = err_mae
    timex_rmses[m] = err_rmse


# In[ ]:


print("########## FINAL RESULTS ##########")
print(f"Case: Covid-19, case: {name}, delta: {delta}")
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


# In[ ]:


with open(f"multivariate/delta_{delta}/historical_predictions.pkl", 'rb') as file:
    p = pickle.load(file)


# In[ ]:


timex_maes = {}
timex_rmses = {}

for i, m in enumerate(p):
    pred_timex = p[m]
    pred_timex = pred_timex[name].astype('float')
    pred_timex = TimeSeries.from_series(pred_timex)
    pred_timex = pred_timex.slice_intersect(backtests[0])
    #     pred_timex = pred_timex.drop_after(backtests[0].time_index()[-1] + pd.Timedelta(days=1))
    assert len(pred_timex) == len(backtests[i])

    err_mae = mae(pred_timex, series)
    err_rmse = rmse(pred_timex, series)

    timex_maes[m] = err_mae
    timex_rmses[m] = err_rmse


# In[ ]:


print("########## FINAL RESULTS ##########")
print(f"Case: Covid-19 - Multivariate, case: {name}, delta: {delta}")
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


# ## Delta 7

# In[2]:


name = 'Daily cases'


# In[3]:


delta = 7


# In[4]:


df = pd.read_csv("Covid19-italy.csv", parse_dates=["Date"], index_col="Date")


# In[5]:


df


# In[6]:


df = df.loc[:pd.Timestamp("20210225"), :]


# In[7]:


df


# In[8]:


series = TimeSeries.from_series(df[name])


# ## Darts

# In[11]:


import functools

backtests = []

for model in models:
    print(f"{model}: running...")
    initial_time = time.time()
    hist_pred = model.historical_forecasts(series,
                            start=pd.Timestamp('2020-08-19'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
    hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
    backtests.append(hist_pred)
    
    final_time = time.time() - initial_time
    print(f"{model}: final time spent: {round(final_time, 3)}")


# In[12]:


from darts.dataprocessing.transformers import Scaler

print(f"LSTM: running...")
initial_time = time.time()
transformer = Scaler()
transformed_series = transformer.fit_transform(series)
lstm = RNNModel(model='LSTM', input_chunk_length=round(len(series)/4), output_chunk_length=1)
models.append(lstm)

hist_pred =  lstm.historical_forecasts(transformed_series,
                            start=pd.Timestamp('2020-08-19'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
backtests.append(transformer.inverse_transform(hist_pred))
final_time = time.time() - initial_time
print(f"LSTM: final time spent: {round(final_time, 3)}")


# In[13]:


darts_maes = {}
darts_rmses = {}

for i, m in enumerate(models):
    prediction = backtests[i]
    #     print(prediction)
    err_mae = mae(backtests[i], series)
    err_rmse = rmse(backtests[i], series)
    darts_maes[m] = err_mae
    darts_rmses[m] = err_rmse

#     print(f"{m}: MAE = {round(err_mae, 3)}, RMSE = {round(err_rmse, 3)}")


# ## Timex

# In[14]:


with open(f"univariate/delta_{delta}/historical_predictions.pkl", 'rb') as file:
    p = pickle.load(file)


# In[15]:


timex_maes = {}
timex_rmses = {}

for i, m in enumerate(p):
    pred_timex = p[m]
    pred_timex = pred_timex[name].astype('float')
    pred_timex = TimeSeries.from_series(pred_timex)
    pred_timex = pred_timex.slice_intersect(backtests[0])
    #     pred_timex = pred_timex.drop_after(backtests[0].time_index()[-1] + pd.Timedelta(days=1))
    assert len(pred_timex) == len(backtests[i])

    err_mae = mae(pred_timex, series)
    err_rmse = rmse(pred_timex, series)

    timex_maes[m] = err_mae
    timex_rmses[m] = err_rmse


# In[16]:


print("########## FINAL RESULTS ##########")
print(f"Case: Covid-19, case: {name}, delta: {delta}")
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


# In[17]:


with open(f"multivariate/delta_{delta}/historical_predictions.pkl", 'rb') as file:
    p = pickle.load(file)


# In[18]:


timex_maes = {}
timex_rmses = {}

for i, m in enumerate(p):
    pred_timex = p[m]
    pred_timex = pred_timex[name].astype('float')
    pred_timex = TimeSeries.from_series(pred_timex)
    pred_timex = pred_timex.slice_intersect(backtests[0])
    #     pred_timex = pred_timex.drop_after(backtests[0].time_index()[-1] + pd.Timedelta(days=1))
    assert len(pred_timex) == len(backtests[i])

    err_mae = mae(pred_timex, series)
    err_rmse = rmse(pred_timex, series)

    timex_maes[m] = err_mae
    timex_rmses[m] = err_rmse


# In[19]:


print("########## FINAL RESULTS ##########")
print(f"Case: Covid-19 - Multivariate, case: {name}, delta: {delta}")
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


# # Daily deaths

# ## Delta 1

# In[ ]:


name = 'Daily deaths'


# In[ ]:


delta = 1


# In[ ]:


df = pd.read_csv("Covid19-italy.csv", parse_dates=["Date"], index_col="Date")


# In[ ]:


df


# In[ ]:


df = df.loc[:pd.Timestamp("20210225"), :]


# In[ ]:


df


# In[ ]:


series = TimeSeries.from_series(df[name])


# ## Darts

# In[ ]:


models = [ExponentialSmoothing(), AutoARIMA(), Prophet()]


# In[ ]:


import functools

backtests = []

for model in models:
    print(f"{model}: running...")
    initial_time = time.time()
    hist_pred = model.historical_forecasts(series,
                            start=pd.Timestamp('2020-08-19'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
    hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
    backtests.append(hist_pred)
    
    final_time = time.time() - initial_time
    print(f"{model}: final time spent: {round(final_time, 3)}")


# In[ ]:


from darts.dataprocessing.transformers import Scaler

print(f"LSTM: running...")
initial_time = time.time()
transformer = Scaler()
transformed_series = transformer.fit_transform(series)
lstm = RNNModel(model='LSTM', input_chunk_length=round(len(series)/4), output_chunk_length=1)
models.append(lstm)

hist_pred =  lstm.historical_forecasts(transformed_series,
                            start=pd.Timestamp('2020-08-19'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
backtests.append(transformer.inverse_transform(hist_pred))
final_time = time.time() - initial_time
print(f"LSTM: final time spent: {round(final_time, 3)}")


# In[ ]:


darts_maes = {}
darts_rmses = {}

for i, m in enumerate(models):
    prediction = backtests[i]
    #     print(prediction)
    err_mae = mae(backtests[i], series)
    err_rmse = rmse(backtests[i], series)
    darts_maes[m] = err_mae
    darts_rmses[m] = err_rmse

    print(f"{m}: MAE = {round(err_mae, 3)}, RMSE = {round(err_rmse, 3)}")


# ## Timex

# In[ ]:


with open(f"univariate/delta_{delta}/historical_predictions.pkl", 'rb') as file:
    p = pickle.load(file)


# In[ ]:


timex_maes = {}
timex_rmses = {}

for i, m in enumerate(p):
    pred_timex = p[m]
    pred_timex = pred_timex[name].astype('float')
    pred_timex = TimeSeries.from_series(pred_timex)
    pred_timex = pred_timex.slice_intersect(backtests[0])
    #     pred_timex = pred_timex.drop_after(backtests[0].time_index()[-1] + pd.Timedelta(days=1))
    assert len(pred_timex) == len(backtests[i])

    err_mae = mae(pred_timex, series)
    err_rmse = rmse(pred_timex, series)

    timex_maes[m] = err_mae
    timex_rmses[m] = err_rmse


# In[ ]:


print("########## FINAL RESULTS ##########")
print(f"Case: Covid-19, case: {name}, delta: {delta}")
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


# In[ ]:


with open(f"multivariate/delta_{delta}/historical_predictions.pkl", 'rb') as file:
    p = pickle.load(file)


# In[ ]:


timex_maes = {}
timex_rmses = {}

for i, m in enumerate(p):
    pred_timex = p[m]
    pred_timex = pred_timex[name].astype('float')
    pred_timex = TimeSeries.from_series(pred_timex)
    pred_timex = pred_timex.slice_intersect(backtests[0])
    #     pred_timex = pred_timex.drop_after(backtests[0].time_index()[-1] + pd.Timedelta(days=1))
    assert len(pred_timex) == len(backtests[i])

    err_mae = mae(pred_timex, series)
    err_rmse = rmse(pred_timex, series)

    timex_maes[m] = err_mae
    timex_rmses[m] = err_rmse


# In[ ]:


print("########## FINAL RESULTS ##########")
print(f"Case: Covid-19 - Multivariate, case: {name}, delta: {delta}")
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


# ## Delta 7

# In[20]:


name = 'Daily deaths'


# In[21]:


delta = 7


# In[22]:


df = pd.read_csv("Covid19-italy.csv", parse_dates=["Date"], index_col="Date")


# In[23]:


df


# In[24]:


df = df.loc[:pd.Timestamp("20210225"), :]


# In[25]:


df


# In[26]:


series = TimeSeries.from_series(df[name])


# ## Darts

# In[27]:


models = [ExponentialSmoothing(), AutoARIMA(), Prophet()]


# In[28]:


import functools

backtests = []

for model in models:
    print(f"{model}: running...")
    initial_time = time.time()
    hist_pred = model.historical_forecasts(series,
                            start=pd.Timestamp('2020-08-19'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
    hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
    backtests.append(hist_pred)
    
    final_time = time.time() - initial_time
    print(f"{model}: final time spent: {round(final_time, 3)}")


# In[29]:


from darts.dataprocessing.transformers import Scaler

print(f"LSTM: running...")
initial_time = time.time()
transformer = Scaler()
transformed_series = transformer.fit_transform(series)
lstm = RNNModel(model='LSTM', input_chunk_length=round(len(series)/4), output_chunk_length=1)
models.append(lstm)

hist_pred =  lstm.historical_forecasts(transformed_series,
                            start=pd.Timestamp('2020-08-19'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
backtests.append(transformer.inverse_transform(hist_pred))
final_time = time.time() - initial_time
print(f"LSTM: final time spent: {round(final_time, 3)}")


# In[30]:


darts_maes = {}
darts_rmses = {}

for i, m in enumerate(models):
    prediction = backtests[i]
    #     print(prediction)
    err_mae = mae(backtests[i], series)
    err_rmse = rmse(backtests[i], series)
    darts_maes[m] = err_mae
    darts_rmses[m] = err_rmse

    print(f"{m}: MAE = {round(err_mae, 3)}, RMSE = {round(err_rmse, 3)}")


# ## Timex

# In[31]:


with open(f"univariate/delta_{delta}/historical_predictions.pkl", 'rb') as file:
    p = pickle.load(file)


# In[32]:


timex_maes = {}
timex_rmses = {}

for i, m in enumerate(p):
    pred_timex = p[m]
    pred_timex = pred_timex[name].astype('float')
    pred_timex = TimeSeries.from_series(pred_timex)
    pred_timex = pred_timex.slice_intersect(backtests[0])
    #     pred_timex = pred_timex.drop_after(backtests[0].time_index()[-1] + pd.Timedelta(days=1))
    assert len(pred_timex) == len(backtests[i])

    err_mae = mae(pred_timex, series)
    err_rmse = rmse(pred_timex, series)

    timex_maes[m] = err_mae
    timex_rmses[m] = err_rmse


# In[33]:


print("########## FINAL RESULTS ##########")
print(f"Case: Covid-19, case: {name}, delta: {delta}")
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


# In[34]:


with open(f"multivariate/delta_{delta}/historical_predictions.pkl", 'rb') as file:
    p = pickle.load(file)


# In[35]:


timex_maes = {}
timex_rmses = {}

for i, m in enumerate(p):
    pred_timex = p[m]
    pred_timex = pred_timex[name].astype('float')
    pred_timex = TimeSeries.from_series(pred_timex)
    pred_timex = pred_timex.slice_intersect(backtests[0])
    #     pred_timex = pred_timex.drop_after(backtests[0].time_index()[-1] + pd.Timedelta(days=1))
    assert len(pred_timex) == len(backtests[i])

    err_mae = mae(pred_timex, series)
    err_rmse = rmse(pred_timex, series)

    timex_maes[m] = err_mae
    timex_rmses[m] = err_rmse


# In[36]:


print("########## FINAL RESULTS ##########")
print(f"Case: Covid-19 - Multivariate, case: {name}, delta: {delta}")
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


# In[ ]:
