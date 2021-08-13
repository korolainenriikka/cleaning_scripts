import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import click


@click.command()
@click.option('--raw_data_location')
@click.option('--wrangled_data_location')
def calculate_statistics(raw_data_location, wrangled_data_location):
    old_df = pd.read_csv(os.path.join(raw_data_location, 'raw_data.csv'))
    new_df = pd.read_csv(os.path.join(wrangled_data_location, 'out.csv'))

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
    
    lat_in_top_bin = sorted(lat_in_top_bin, key=lambda x: x[0])
    lon_in_top_bin = sorted(lon_in_top_bin, key=lambda x: x[0])

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,20))
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

    plotfilename = title.replace(' ', '_') + '.png'
    plt.savefig(plotfilename)
    mlflow.log_artifact(plotfilename)

if __name__ == '__main__':
    calculate_statistics()
