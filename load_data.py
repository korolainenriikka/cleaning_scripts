import logging
import pandas as pd

# step params to parse: filepath, id_column, lon_column, lat_column, time_column
# then: load data into artifacts

def load_dataset():
    '''Loads the dataset. Then removes all id/time duplicates and rows with missing coordinates. Return dataframe with renamed columns '''
    logging.info("+++ START +++\n+++ READING DATAFRAME FROM FILE +++")
    df = pd.read_csv(filepath)
    original_length = len(df)
    logging.info("+++ ORIGINAL LENGTH: " + str(original_length) + "+++\n")
    logging.info("+++ DROPPING ID/TIME DUPLICATES +++")
    df = df.drop_duplicates([time_column, id_column])
    prev_len = len(df)
    logging.info("+++ ROWS DROPPED: " + str(original_length-prev_len) + "+++\n")
    # Standardise column names
    df = df.rename(columns={id_column:'id', lon_column: 'lon', lat_column: 'lat', time_column:'t'})
    # Drop rows that are missing coordinates
    logging.info("+++ DROPPING ROWS WHERE COORDINATES ARE MISSING +++")
    prev_len = len(df)
    df = df.dropna(subset=['lon', 'lat'], how='any')
    logging.info("+++ ROWS DROPPED:" + str(prev_len-len(df)) + "+++\n")
    return df
