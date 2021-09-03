#!/usr/bin/env bash

echo "Creating Python virtual environment..."
python -m venv venv_timex
source venv_timex/bin/activate

echo "Install Torch CPU (to be safe). Comment this if you prefer Torch with CUDA."
pip install wheel
pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install poetry

echo "Installing TIMEX..."
poetry install
pip install typing_extensions
