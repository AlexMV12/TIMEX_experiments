#!/usr/bin/env bash

source venv_timex/bin/activate

cd Studio-Covid19/univariate/delta_1
rm historical_predictions.pkl
python -c 'from app import compute; compute()'

cd ../delta_7
rm historical_predictions.pkl
python -c 'from app import compute; compute()'

cd ../../multivariate/delta_1
rm historical_predictions.pkl
python -c 'from app import compute; compute()'

cd ../delta_7
rm historical_predictions.pkl
python -c 'from app import compute; compute()'

cd ../../..
deactivate
source venv_darts/bin/activate
cd Studio-Covid19
python run_darts.py
