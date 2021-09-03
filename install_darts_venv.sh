#!/usr/bin/env bash

echo "Installing Python virtual environment with Darts..."

python -m venv venv_darts
source venv_darts/bin/activate
pip install wheel
pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install darts
