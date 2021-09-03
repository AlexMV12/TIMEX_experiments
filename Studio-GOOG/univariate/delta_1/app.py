import json
import logging
import os
import pickle
import sys
import webbrowser
import dateparser
import numpy

from pandas import read_csv, DataFrame

import timexseries.data_ingestion
from timexseries.data_prediction.xcorr import calc_xcorr

from timexseries.data_ingestion import add_freq, select_timeseries_portion, add_diff_columns
from timexseries.data_prediction.models.prophet_predictor import FBProphetModel
from timexseries.data_prediction import create_timeseries_containers
from timexseries.timeseries_container import TimeSeriesContainer

log = logging.getLogger(__name__)


def compute():

    param_file_nameJSON = 'configurations/configuration_test_finance.json'

    # Load parameters from config file.
    with open(param_file_nameJSON) as json_file:  # opening the config_file_name
        param_config = json.load(json_file)  # loading the json

    # Logging
    log_level = getattr(logging, param_config["verbose"], None)
    if not isinstance(log_level, int):
        log_level = 0
    # %(name)s for module name
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level, stream=sys.stdout)

    # data ingestion
    log.info(f"Started data ingestion.")
    ingested_data = timexseries.data_ingestion.ingest_timeseries(param_config)  # ingestion of data

    # data selection
    log.info(f"Started data selection.")
    ingested_data = select_timeseries_portion(ingested_data, param_config)

    # data prediction
    containers = create_timeseries_containers(ingested_data=ingested_data, param_config=param_config)

    # Save the computed data; these are the TimeSeriesContainer objects from which a nice Dash page can be built.
    # They can be loaded by "app_load_from_dump.py" to start the app
    # without re-computing all the data.
    with open(f"containers.pkl", 'wb') as input_file:
        pickle.dump(containers, input_file)


if __name__ == '__main__':
    compute()


    def open_browser():
        webbrowser.open("http://127.0.0.1:8000")


    # Timer(6, open_browser).start()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.system("gunicorn -b 0.0.0.0:8003 app_load_from_dump:server")


