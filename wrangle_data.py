import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


def drop_small_legs(df, leg_gap, min_rows):
    ''' If timedelta between two measurements is more than the threshold defined in config file  '''
    interm = df[(df.timedelta > leg_gap) | df.timedelta.isna()].reset_index()['index']
    df['leg_num'] = pd.Series(interm.index, index=interm)
    df['leg_num'] = df.groupby('id').leg_num.fillna(method='ffill')
    logging.info("+++ DROPPING SHIPS THAT HAVE FEWER THAN " + min_rows + " MEASUREMENTS +++")
    prev_len = len(df)
    df = df[df.groupby('id')['id'].transform('size') > min_rows]
    logging.info("+++ ROWS DROPPED: " + str(prev_len-len(df)) + " +++\n")
    return df


def fix_values(df, max_speed, fix_values_bool):
    if fix_values_bool:
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
        df = df[df['speed'] < max_speed]
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
            df['t'] = df['t'].apply(lambda x: df.parse(x))
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


def wrangle_dataset():
    # Bearing calculation function
    def calculate_bearing(lat):
        dlon = np.absolute(df.loc[lat.index, 'lon'] - df.loc[lat.index, 'lon_prev'])
        X = np.cos(df.loc[lat.index, 'lat_prev'])* np.sin(dlon)
        Y = np.cos(lat) * np.sin(df.loc[lat.index, 'lat_prev']) - np.sin(lat) * np.cos(df.loc[lat.index, 'lat_prev']) * np.cos(dlon)
        return np.degrees(np.arctan2(X,Y))

    # TODO: read dataframe from artifacts store
    # read params from mlflow run command

    df = pd.DataFrame()

    min_rows = 20
    max_speed = 25
    leg_gap = 1800
    fix_values_param = 0
    fix_values_bool = bool(fix_values_param)

    original_length=len(df)
    # Calculate timedeltas
    df = timedelta(df)
    # Differentiate the legs
    df = drop_small_legs(df)
    # Remove rows where the speed is above limit
    
    df = calculate_speed(df)

    df = fix_values(df, fix_values_bool)
    '''
    if bearing:
        df['bearing'] = df.groupby('leg_num')['lat'].apply(calculate_bearing)
    '''
    logging.info("+++ TOTAL ROWS DROPPED: " + str(original_length-len(df)) + " +++")
    logging.info("---DONE---")
    return df
