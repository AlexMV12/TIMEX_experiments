import json
import logging
import pickle

import dash
import dash_html_components as html
from datetime import datetime, timezone
import dash_core_components as dcc
import dateparser
from dash.dependencies import Input, Output
from pandas import read_csv

from timexseries.data_ingestion import add_diff_columns
from timexseries.data_visualization.functions import create_timeseries_dash_children, line_plot_multiIndex

log = logging.getLogger(__name__)


param_file_nameJSON = 'configurations/configuration.json'

# Load parameters from config file.
with open(param_file_nameJSON) as json_file:  # opening the config_file_name
    param_config = json.load(json_file)  # loading the json

# Load containers dump.
with open(f"containers.pkl", 'rb') as input_file:
    timeseries_containers = pickle.load(input_file)

# Data visualization
children_for_each_timeseries = [{
    'name': s.timeseries_data.columns[0],
    'children': create_timeseries_dash_children(s, param_config)
} for s in timeseries_containers]


# Initialize Dash app.
app = dash.Dash(__name__)
server = app.server

now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

disclaimer = [html.Div([
    html.H1("TIMEX for Milk production", style={'text-align': 'center'}),
    html.Hr(),
    dcc.Markdown('''
        TIMEX results.
        '''),
    html.Div("Last updated at (yyyy-mm-dd, UTC time): " + str(now)),
    html.Br(),
    html.H2("Please select the data of interest:")
], style={'width': '80%', 'margin': 'auto'}
), dcc.Dropdown(
    id='timeseries_selector',
    options=[{'label': i['name'], 'value': i['name']} for i in children_for_each_timeseries],
    value='Time-series'
), html.Div(id="timeseries_wrapper"), html.Div(dcc.Graph(), style={'display': 'none'})]
tree = html.Div(children=disclaimer, style={'width': '80%', 'margin': 'auto'})

app.layout = tree


@app.callback(
    Output(component_id='timeseries_wrapper', component_property='children'),
    [Input(component_id='timeseries_selector', component_property='value')]
)
def update_timeseries_wrapper(input_value):
    try:
        children = next(x['children'] for x in children_for_each_timeseries if x['name'] == input_value)
    except StopIteration:
        return html.Div(style={'padding': 200})

    return children


