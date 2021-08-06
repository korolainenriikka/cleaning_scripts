# Tool for data cleaning

This tool is meant to be used to clean AIS data in a machine learning pipeline. The script removes all time/id duplicates from the data and all the rows that have missing coordinate values. It also calculates the speeds for each time entry based on previous coordinates and difference in time. 


### Usage

Run this script from the command line by command `python3 wrangle_script.py <filepath>`. Filepath is the path to the file to be wrangled

This script expects that the dataframe will have columns named 'id', 'lat', 'lon' and 't'

If the columns are named differently you can input the alternative names using --column column_in_df syntax

So for example if the dataframe would have id in a column called mmsi you would run the script with

`python3 wrangle_script.py \<filepath\> --id mmsi`

Also flags -b and -f are used to further wrangle the data. Flag -b calculates the bearing in each point and -f flag does not remove the outliers from the data and instead imputs values using rolling average method.


### Example use


Download dataset from https://zenodo.org/record/3754481

Use script with command 

`python3 wrangle_script.py <filepath> --id shipid`

