#!/usr/bin/env bash

source venv_timex/bin/activate
cd Studio-Monthly-Milk-Production/univariate/delta_1
rm historical_predictions.pkl
python -c 'from app import compute; compute()'

cd ../delta_3
rm historical_predictions.pkl
python -c 'from app import compute; compute()'

cd ../delta_6
rm historical_predictions.pkl
python -c 'from app import compute; compute()'

cd ../../..
deactivate
source venv_darts/bin/activate
cd Studio-Monthly-Milk-Production
python run_darts.py
