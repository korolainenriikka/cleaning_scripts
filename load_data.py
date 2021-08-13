import logging
import pandas as pd
import numpy as np
import mlflow
import click

@click.command()
@click.option('--filepath')
@click.option('--id_column')
@click.option('--lon_column')
@click.option('--lat_column')
@click.option('--time_column')
def load_dataset(filepath, id_column, lon_column, lat_column, time_column):
    with mlflow.start_run():       
        '''Loads the dataset. Then removes all id/time duplicates and rows with missing coordinates. Return dataframe with renamed columns '''
        logging.info("+++ START +++\n+++ READING DATAFRAME FROM FILE +++")
        df = pd.read_csv(filepath)
        original_length = len(df)
        logging.info("+++ ORIGINAL LENGTH: " + str(original_length) + "+++\n")
        mlflow.log_metric('original length', original_length)

        logging.info("+++ DROPPING ID/TIME DUPLICATES +++")
        df = df.drop_duplicates([time_column, id_column])
        prev_len = len(df)
        dropped_rows = original_length - prev_len
        logging.info("+++ ROWS DROPPED: " + str(dropped_rows) + "+++\n")
        mlflow.log_metric('count of id/time duplicates', dropped_rows)
        
        # Standardise column names
        df = df.rename(columns={id_column:'id', lon_column: 'lon', lat_column: 'lat', time_column:'t'})
        
        # Drop rows that are missing coordinates
        logging.info("+++ DROPPING ROWS WHERE COORDINATES ARE MISSING +++")
        prev_len = len(df)
        df = df.dropna(subset=['lon', 'lat'], how='any')
        coordinates_missing_count = prev_len-len(df)
        logging.info("+++ ROWS DROPPED:" + str(coordinates_missing_count) + "+++\n")
        mlflow.log_metric('count of entries with missing coordinates', coordinates_missing_count)
        
        # store read data into artifacts
        df.to_csv('raw_data.csv')
        mlflow.log_artifact('raw_data.csv')

if __name__ == '__main__':
    load_dataset()
