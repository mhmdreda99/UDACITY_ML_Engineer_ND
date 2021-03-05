import json
import os


def make_time_series(ts):
    '''Generate time series for test model.
    
    Paramenters
    -----------
    ts : series from pandas
    
    Returns
    -------
    A list of time series.
    '''
    
    idx = [20, 15, 10, 5]
    time_series = []
    for i in idx:
        time_series.append(ts.iloc[:-i])
    return time_series


def create_training_series(complete_time_series, prediction_length):
    '''Given a complete list of time series data, create training time series.
       
    Paramenters
    -----------
    complete_time_series : A list of all complete time series.
    prediction_length (int) : The number of points we want to predict.
    
    Returns
    -------
    A list of training time series.
    '''
    
    # get training series
    time_series_training = []
    for ts in complete_time_series:
        time_series_training.append(ts[:-prediction_length])
    return time_series_training


def series_to_json_obj(ts):
    '''Returns a dictionary of values in DeepAR, JSON format.
    
    Parameters
    ----------
    ts : A single time series.
       
    Returns
    -------
    A dictionary of values with "start" and "target" keys.
    '''
    
    # get start time and target from the time series, ts
    json_obj = {'start': str(ts.index[0]), 'target': list(ts)}
    return json_obj
    
    
def write_json_dataset(time_series, filename):
    '''Write time series in json format.
    
    Parameters
    ----------
    time_series : list of time series
    filename : file path
    
    Returns
    -------
    Json with time series ready to use in DeepAR model
    '''
    
    with open(filename, 'wb') as f:
        # for each of our times series, there is one JSON line
        for ts in time_series:
            json_line = json.dumps(series_to_json_obj(ts)) + '\n'
            json_line = json_line.encode('utf-8')
            f.write(json_line)
    print(filename + ' saved.')
    
    
