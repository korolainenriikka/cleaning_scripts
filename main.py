import mlflow
import logging
from configparser import ConfigParser
import argparse


def run_step(entrypoint, parameters=None):
    print("----------\nLAUNCHING STEP: entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


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

        dataframe_args = dict(config_object['DATAFRAME'])
        filepath, id_column, lon_column, lat_column, time_column = dataframe_args.values()

        # MLflow does not allow passing objects as run params
        submitted_load_data = run_step('load_data', parameters={
            'filepath': filepath,
            'id_column': id_column,
            'lon_column': lon_column,
            'lat_column': lat_column,
            'time_column': time_column,
        })
        data_location = submitted_load_data.info.artifact_uri

        wrangling_args = dict(config_object['WRANGLING'])
        min_rows, max_speed, leg_gap, fix_values = wrangling_args.values()

        # note on params: mlflow has no boolean param so 0 is false 1 is true
        submitted_wrangle_data = run_step('wrangle_data', parameters={
            'data_location': data_location,
            'min_rows': min_rows,
            'max_speed': max_speed,
            'leg_gap': leg_gap,
            'fix_values': str(fix_values)
        })
        wrangled_location = submitted_wrangle_data.info.artifact_uri


        run_step('calculate_statistics', parameters={'raw_data_location': data_location, 'wrangled_data_location': wrangled_location})
        print('workflow finished.')

if __name__ == '__main__':
    main()
