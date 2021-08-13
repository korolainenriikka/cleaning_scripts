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

### On the mlflow workflow

### Run the workflow

* Run this mlflow workflow by running `mlflow run .` in the project root
* In order to see the logs made by MLflow in the UI, run `mlflow ui` in the path where the `mlruns` directory is located (it is created to the location where `mlflow run` is executed)
    * to see individual workflow steps in the ui, click the + icon next to the full workflow run.   

### What does ... do/mean?

* click is used to parse options given to each steps' `mlflow run` command (in MLProject file). This may or may not be the best library for this use, MLflow used it in their multistep example
* the return value of `run_step` in `main.py` is used to locate artifacts stored in previous step to read them in the next one. This data passing to another step could also be implemented by eg. passing a database uri as a param.
