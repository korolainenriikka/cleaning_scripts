import pandas as pd
import numpy as np
import dateutil.parser as dp
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import sys
import argparse
import matplotlib.pyplot as plt
import logging
from configparser import ConfigParser
import mlflow


def calculate_statistics(old_df, new_df):
    print("+++ CALCULATING STATISTICS +++")
    print("+++ ORIGINAL DATAFRAME +++")
    print(old_df.describe())
    print("+++ WRANGLED DATAFRAME +++")
    print(new_df.describe())
    show_percentage_of_rows_with_zero_diff_per_leg(old_df, 'id', "Raw dataframe groupby id")
    show_percentage_of_rows_with_zero_diff_per_leg(new_df, 'id', "Wrangled dataframe groupby id")
    show_percentage_of_rows_with_zero_diff_per_leg(new_df, 'leg_num', 'Wrangled dataframe groupby leg id')

def show_percentage_of_rows_with_zero_diff_per_leg(df, groupby_id, title):
    df = df.sort_values('t')
    df['dlat'] = df.groupby(groupby_id).lat.diff()
    df['dlon'] = df.groupby(groupby_id).lon.diff()
    lat_in_top_bin = []
    lon_in_top_bin = []
    for g_id in df[groupby_id].unique():
        d = df[df[groupby_id]==g_id].dropna(subset=['lat', 'lon'], how='any')
        len_lat = len(d.dlat)
        len_lon = len(d.dlon)
        if len_lat < 1 or len_lon < 1:
            continue
        lat_in_top_bin.append((len_lat, len(d[d.dlat!=0.0])/len_lat))
        lon_in_top_bin.append((len_lon, len(d[d.dlon!=0.0])/len_lon))
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,20))
    lat_in_top_bin = sorted(lat_in_top_bin, key=lambda x: x[0])
    lon_in_top_bin = sorted(lon_in_top_bin, key=lambda x: x[0])
    fig.suptitle(title)
    ax1.set_xlabel('number of records')
    ax1.set_title('Delta latitude')
    ax1.set_ylabel('percentage of zero diff rows')
    ax2.set_xlabel('number of records')
    ax2.set_title('Delta longitude')
    ax2.set_ylabel('percentage of zero diff rows')
    a = np.array(lat_in_top_bin).T
    b = np.array(lon_in_top_bin).T
    ax1.scatter(a[0], a[1]*100, color="red")
    ax2.scatter(b[0], b[1]*100, color="blue")
    #plt.show()
    mlflow.log_artifact(plt)

def load_dataset():
    '''Loads the dataset. Then removes all id/time dublicates and rows with missing coordinates. Return dataframe with renamed columns '''
    dataframe_args = dict(config_object['DATAFRAME'])
    filepath, id_column, lon_column, lat_column, time_column = dataframe_args.values()
    logging.info("+++ START +++\n+++ READING DATAFRAME FROM FILE +++")
    df = pd.read_csv(filepath)
    original_length = len(df)
    logging.info("+++ ORIGINAL LENGTH: " + str(original_length) + "+++\n")
    logging.info("+++ DROPPING ID/TIME DUBLICATES +++")
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

def drop_small_legs(df):
    ''' If timedelta between two measurements is more than the threshold defined in config file  '''
    interm = df[(df.timedelta > int(config_object['WRANGLING']['leg_gap'])) | df.timedelta.isna()].reset_index()['index']
    df['leg_num'] = pd.Series(interm.index, index=interm)
    df['leg_num'] = df.groupby('id').leg_num.fillna(method='ffill')
    logging.info("+++ DROPPING SHIPS THAT HAVE FEWER THAN " + config_object['WRANGLING']['min_rows'] + " MEASUREMENTS +++")
    prev_len = len(df)
    df = df[df.groupby('id')['id'].transform('size')>int(config_object['WRANGLING']['min_rows'])]
    logging.info("+++ ROWS DROPPED: " + str(prev_len-len(df)) + " +++\n")
    return df

def fix_values(df):
    if bool(config_object['WRANGLING']['fix_values']):
        ship_legs = df.groupby('leg_num')
        logging.info("+++ FIXING SPEED VALUES OUTLIERS +++")
        df.lon = np.where(df['speed']>25, np.nan, df.lon)
        df.lat = np.where(df['speed']>25, np.nan, df.lat)
        logging.info("+++ TOTAL OF OUTLIERS: LAT-" + str(len(df[df.lat.isna()])) + " LON-" + str(len(df[df.lon.isna()])))
        rolling_averages_lat = ship_legs.lat.rolling(10, min_periods=1, center=True).mean()
        rolling_averages_lon = ship_legs.lon.rolling(10, min_periods=1, center=True).mean()
        df.lat.fillna(rolling_averages_lat.reset_index(level=0)['lat'])
        df.lon.fillna(rolling_averages_lon.reset_index(level=0)['lon'])
        return df
    else:
        logging.info("+++ DROPPING SPEED VALUE OUTLIERS +++")
        df = df[df['speed']<int(config_object['WRANGLING']['max_speed'])]
        return df

def timedelta(df):
    '''Method for calculating the timedelta between each entry within a ship'''
    if is_numeric_dtype(df['t']):
        # Sort values by timestamp
        df = df.sort_values('t')
        ships  = df.groupby('id')
        df['timedelta'] = ships['t'].diff()
    else: 
        # Change time to datetime if not already in this format
        if not is_datetime64_any_dtype(df['t']):
            df['t'] = df['t'].apply(lambda x: dp.parse(x))
        # Sort values by timestamp
        df = df.sort_values('t')
        ships  = df.groupby('id')
        df['timedelta'] = ships['t'].diff().transform(lambda x: abs(x.total_seconds()))
    return df

def calculate_speed(df):
    '''Calculate speed using timedeltas and haversine distance between two points. Uses auxiliarry calculate_speed function for fast axis wise application'''
    # Auxiliary function
    def calculate_speed_aux(timedelta):
        lat1, lon1, lat2, lon2 = map(np.radians, [df.loc[timedelta.index, 'lat'], df.loc[timedelta.index, 'lon'], df.loc[timedelta.index, 'lat_prev'], df.loc[timedelta.index, 'lon_prev']])
        # haversine formula
        dlon = lon2 - lon1 
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 
        r = 6371000 # Radius of earth in meters. Use 3956 for miles
        mps_to_knots = 1.944 # 1 meter per second is about 1.944 knots 
        return mps_to_knots*((c * r) / timedelta)
    ships = df.groupby('id')
    df['lon_prev'] = ships.lon.shift(1)
    df['lat_prev'] = ships.lat.shift(1)
    df['speed'] = ships.timedelta.apply(calculate_speed_aux)
    return df

def wrangle_dataset(df):
    # Bearing calculation function
    def calculate_bearing(lat):
        dlon = np.absolute(df.loc[lat.index, 'lon'] - df.loc[lat.index, 'lon_prev'])
        X = np.cos(df.loc[lat.index, 'lat_prev'])* np.sin(dlon)
        Y = np.cos(lat) * np.sin(df.loc[lat.index, 'lat_prev']) - np.sin(lat) * np.cos(df.loc[lat.index, 'lat_prev']) * np.cos(dlon)
        return np.degrees(np.arctan2(X,Y))

    original_length=len(df)
    # Calculate timedeltas
    df = timedelta(df)
    # Differentiate the legs
    df = drop_small_legs(df)
    # Remove rows where the speed is above limit
    
    df = calculate_speed(df)

    df = fix_values(df)
    '''
    if bearing:
        df['bearing'] = df.groupby('leg_num')['lat'].apply(calculate_bearing)
    '''
    logging.info("+++ TOTAL ROWS DROPPED: " + str(original_length-len(df)) + " +++")
    logging.info("---DONE---")
    return df


def main():
    with mlflow.start_run():
        parser = argparse.ArgumentParser(description="Tool for data wrangling")
        parser.add_argument('-v', '--verbose', help = 'show program output', default=False, action="store_true")
        parser.add_argument('-s', '--statistics', help='calculate statistics on the transformation', default=False, action="store_true")

        config_object = ConfigParser()
        config_object.read("config.ini")
        cli_args=vars(parser.parse_args())
        if cli_args['verbose']:
            logging.basicConfig(level=logging.DEBUG)

        raw_dataframe = load_dataset()

        fixed_dataframe = wrangle_dataset(raw_dataframe)

        if cli_args['statistics']:
            calculate_statistics(raw_dataframe, fixed_dataframe)
        fixed_dataframe.to_csv("out_" + config_object['DATAFRAME']['filepath'])

if __name__ == '__main__':
    main()
