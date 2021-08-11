
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

        raw_dataframe = load_dataset(config_object)

        fixed_dataframe = wrangle_dataset(raw_dataframe, config_object=config_object)

        if cli_args['statistics']:
            calculate_statistics(raw_dataframe, fixed_dataframe)
        fixed_dataframe.to_csv("out_" + config_object['DATAFRAME']['filepath'])

if __name__ == '__main__':
    main()
