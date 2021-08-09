import pandas as pd
import numpy as np
import dateutil.parser as dp
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import sys
import argparse
import matplotlib.pyplot as plt

def calculate_statistics(df):
    print("+++ CALCULATING STATISTICS +++")
    df = df.sort_values('t')
    df['dlat'] = df.groupby('id').lat.diff()
    df['dlon'] = df.groupby('id').lon.diff()
    lat_in_top_bin = []
    lon_in_top_bin = []
    for shipid in df['id'].unique():
        d = df[df.id==shipid].dropna(subset=['lat', 'lon'], how='any')
        len_lat = len(d.dlat)
        len_lon = len(d.dlon)
        if len_lat < 1 or len_lon < 1:
            continue
        lat_in_top_bin.append((len_lat, len(d[d.dlat!=0.0])/len_lat))
        lon_in_top_bin.append((len_lon, len(d[d.dlon!=0.0])/len_lon))
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,20))
    lat_in_top_bin = sorted(lat_in_top_bin, key=lambda x: x[0])
    lon_in_top_bin = sorted(lon_in_top_bin, key=lambda x: x[0])
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
    plt.show()

def wrangle_dataset(filepath, id='id', lon='lon', lat='lat', time='t', fix=False, bearing=False):
    # Load dataframe
    print("+++ START +++\n")
    print("+++ READING DATAFRAME FROM FILE +++")
    df = pd.read_csv(filepath)
    original_length = len(df)
    
    print("+++ ORIGINAL LENGTH: ", original_length, "+++\n")
    
    # Speed calculation function
    def calculate_speed(timedelta):
        lat1, lon1, lat2, lon2 = map(np.radians, [df.loc[timedelta.index, 'lat'], df.loc[timedelta.index, 'lon'], df.loc[timedelta.index, 'lat_prev'], df.loc[timedelta.index, 'lon_prev']])
        # haversine formula
        dlon = lon2 - lon1 
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 
        r = 6371000 # Radius of earth in meters. Use 3956 for miles
        mps_to_knots = 1.944 # 1 meter per second is about 1.944 knots 
        return mps_to_knots*((c * r) / timedelta)
    
    # Bearing calculation function
    def calculate_bearing(lat):
        dlon = np.absolute(df.loc[lat.index, 'lon'] - df.loc[lat.index, 'lon_prev'])
        X = np.cos(df.loc[lat.index, 'lat_prev'])* np.sin(dlon)
        Y = np.cos(lat) * np.sin(df.loc[lat.index, 'lat_prev']) - np.sin(lat) * np.cos(df.loc[lat.index, 'lat_prev']) * np.cos(dlon)
        return np.degrees(np.arctan2(X,Y))
    
    # Drop time/id dublicates
    print("+++ DROPPING ID/TIME DUBLICATES +++")
    df = df.drop_duplicates([time, id])
    prev_len = len(df)
    print("+++ ROWS DROPPED:", original_length-prev_len, "+++\n")
    # Standardise column names
    df = df.rename(columns={id:'id', lon: 'lon', lat: 'lat', time:'t'})
    # Drop rows that are missing coordinates
    print("+++ DROPPING ROWS WHERE COORDINATES ARE MISSING +++")
    df = df.dropna(subset=['lon', 'lat'], how='any')
    print("+++ ROWS DROPPED:", prev_len- len(df), "+++\n")
    prev_len=len(df)
    # Calculate timedeltas
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
    
    # Differentiate the legs
    interm = df[(df.timedelta > (30*60)) | df.timedelta.isna()].reset_index()['index']
    df['leg_num'] = pd.Series(interm.index, index=interm)
    df['leg_num'] = df.groupby('id').leg_num.fillna(method='ffill')
    # Calculate speeds
    df['lon_prev'] = ships.lon.shift(1)
    df['lat_prev'] = ships.lat.shift(1)
    df['speed'] = df.groupby('leg_num')['timedelta'].apply(calculate_speed)
    if bearing:
        df['bearing'] = df.groupby('leg_num')['lat'].apply(calculate_bearing)
    if fix:
        print("+++ FIXING SPEED VALUES OUTLIERS +++")
        df.lon = np.where(df['speed']>25, np.nan, df.lon)
        df.lat = np.where(df['speed']>25, np.nan, df.lat)
        print("+++ TOTAL OF OUTLIERS: LAT-", len(df[df.lat.isna()]), "LON-", len(df[df.lon.isna()]))
        rolling_averages_lat = ships.lat.rolling(10, min_periods=1, center=True).mean()
        rolling_averages_lon = ships.lon.rolling(10, min_periods=1, center=True).mean()
        df.lat.fillna(rolling_averages_lat.reset_index(level=0)['lat'])
        df.lon.fillna(rolling_averages_lon.reset_index(level=0)['lon'])
        return df
    # Do the maskings for outlier speeds
    print("+++ DROPPING SPEED VALUE OUTLIERS +++")
    df = df[df['speed']<25]
    print("+++ ROWS DROPPED:", prev_len - len(df),"+++\n")
    print("+++ TOTAL ROWS DROPPED:", original_length-len(df), "+++")
    print("---DONE---")
    return df

parser = argparse.ArgumentParser(description="Tool for data wrangling")
parser.add_argument('filepath', help='filepath of data to be processed')
parser.add_argument('-i','--id', help='name of id column if other than "id"', default='id')
parser.add_argument('-x','--lon', help='name of lon column if other than "lon"', default='lon')
parser.add_argument('-y','--lat', help='name of lat column if other than "lat"',default='lat')
parser.add_argument('-t','--time', help='name of time column if other than "time"', default='t')
parser.add_argument('-f','--fix', help='use flag to fix outlier values using rolling average', default=False, action="store_true")
parser.add_argument('-b', '--bearing', help='use flag to calculate bearing', default=False, action="store_true")
parser.add_argument('-s', '--statistics', help='use flag to calculate statistics', default=False, action="store_true")
args=vars(parser.parse_args())
statistics = args.pop('statistics')
df = wrangle_dataset(**args)
if statistics:
    calculate_statistics(df)
df.to_csv("out_" + args['filepath'])
