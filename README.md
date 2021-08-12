# Tool for data cleaning

This tool is meant to be used to clean AIS data in a machine learning pipeline. The script removes all time/id duplicates from the data and all the rows that have missing coordinate values. It also differentiates distict legs based when there is a timegap in the data. After this it calculates the speeds for each time entry based on previous coordinates and difference in time. 


### Usage

Run this script from the command line by command `python3 wrangle_script.py`. Define the filepath and column names in the config.ini file

The config file also takes some parameters for the wrangling:

- min_rows : the minimum number of rows allowed for a given leg. Legs with fewer rows than this are removed from the data set.
- max_speed : the maximum speed allowed for a ship. Faster ships than this are considered outliers and aree removed from the data set. 
- leg_gap : the time gap needed to differentiate a new leg in seconds
- fix_values : if True fix the coordinates of the rows with outlier speed based on rolling average. Otherwise remove them

The script itself can be used with two flags:
- -v to print out the program output
- -s to calculate statistics on the transformation


### Example use


Download dataset from https://zenodo.org/record/3754481 and unzip the file 'ais.csv' to project root

Use script with command 

`python3 wrangle_script.py` 

If using a different dataset change the `config.ini` file accordingly


### The mlflow adaptation

todo:
append statistics generation into the pipeline
log wrangle script output metrics as mlflow metrics
