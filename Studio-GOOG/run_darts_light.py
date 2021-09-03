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


name = 'GOOG'
delta = 7


# In[86]:


df = pd.read_csv("Stocks.csv", parse_dates=["Date"], index_col="Date")


# In[87]:


df


# In[88]:


df = df.loc[pd.Timestamp("20200101"):pd.Timestamp("20210101"), :]


# In[89]:


df


# In[90]:


series = TimeSeries.from_series(df[name])


# ## Darts

# In[91]:


models = [ExponentialSmoothing(), AutoARIMA(), Prophet()]


# In[92]:


import functools

backtests = []

for model in models:
    print(f"{model}: running...")
    initial_time = time.time()
    hist_pred = model.historical_forecasts(series,
                            start=pd.Timestamp('2020-10-01'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
    hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
    backtests.append(hist_pred)
    
    final_time = time.time() - initial_time
    print(f"{model}: final time spent: {round(final_time, 3)}")


# In[93]:


from darts.dataprocessing.transformers import Scaler

print(f"LSTM: running...")
initial_time = time.time()
transformer = Scaler()
transformed_series = transformer.fit_transform(series)
lstm = RNNModel(model='LSTM', input_chunk_length=round(len(series)/4), output_chunk_length=1)
models.append(lstm)

hist_pred =  lstm.historical_forecasts(transformed_series,
                            start=pd.Timestamp('2020-10-01'),
                            forecast_horizon=delta, stride=delta, verbose=True, last_points_only=False)
hist_pred = functools.reduce(lambda a, b: a.append(b), hist_pred)
backtests.append(transformer.inverse_transform(hist_pred))
final_time = time.time() - initial_time
print(f"LSTM: final time spent: {round(final_time, 3)}")


# In[94]:


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

# In[95]:


with open(f"univariate/delta_{delta}/historical_predictions.pkl", 'rb') as file:
    p = pickle.load(file)


# In[96]:


timex_maes = {}
timex_rmses = {}

for i, m in enumerate(p):
    pred_timex = p[m]
    pred_timex = pred_timex[name].astype('float')
    pred_timex = TimeSeries.from_series(pred_timex)
    pred_timex = pred_timex.slice_intersect(backtests[0])
    #     pred_timex = pred_timex.drop_after(backtests[0].time_index()[-1] + pd.Timedelta(days=1))
    assert len(pred_timex) == len(backtests[0])

    err_mae = mae(pred_timex, series)
    err_rmse = rmse(pred_timex, series)

    timex_maes[m] = err_mae
    timex_rmses[m] = err_rmse


# In[97]:


print("########## FINAL RESULTS ##########")
print(f"Case: GOOG, delta: {delta}")
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


# In[41]:




