################################################################################################
# Author: Joan Ponsa
# Created on: 29/10/2023
# Last modified: 30/10/2023
#
# Downloads the original UCI heart diseases dataset and joins the data in a single file
#
################################################################################################

# Import libraries
from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd

# Set Working directory
load_dotenv()
WORKING_DIR = os.getenv("WORKING_DIR")
os.chdir(WORKING_DIR)

# Create Data directory if doesn't exist
if not os.path.isdir(WORKING_DIR + "/Data/"):
    os.mkdir(WORKING_DIR + "/Data/")

# Create Origila_dataset directory if doesn't exist
if not os.path.isdir(WORKING_DIR + "/Data/Original_dataset/"):
    os.mkdir(WORKING_DIR + "/Data/Original_dataset/")
    print(WORKING_DIR + "/Data/Original_dataset/ created")

# Download data from UCI
if not os.path.isfile(WORKING_DIR + "/Data/Original_dataset/heart+disease.zip"):
    # Download file
    import urllib.request

    url = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
    path_to_zip_file = WORKING_DIR + "/Data/Original_dataset/heart+disease.zip"
    urllib.request.urlretrieve(url, path_to_zip_file)

    # Unizip file
    import zipfile

    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(WORKING_DIR + "/Data/Original_dataset/")

# table field names from original dataset
column_names = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Data was collected in from 4 locations: Cleveland, Hungary, Switzerland and Long Beach VA
# See Data/Original_dataset/heart-diease.names

data_centers = ["cleveland", "hungarian", "switzerland", "va"]

# Combine all 4 datasets in a single file
df = pd.DataFrame([], columns=["data_center"] + column_names)

for center in data_centers:
    # Read the original dataset
    center_file = f"Data/Original_dataset/processed.{center}.data"
    sep = ","
    ## If reprocessed file exist use it instead of processed
    if os.path.isfile(WORKING_DIR + f"Data/Original_dataset/reprocessed.{center}.data"):
        center_file = f"Data/Original_dataset/reprocessed.{center}.data"
        sep = " "

    tmp = pd.read_csv(WORKING_DIR + center_file, header=None, sep=sep)

    # Add the column names
    tmp.columns = column_names
    # Add a column with the data center name (source of the data)
    tmp.insert(loc=0, column="data_center", value=center)
    # Concatenate all in a single final file
    df = pd.concat([df, tmp])

# missing values are encoded using "?". Replace them with NaN
df.replace("?", np.nan, inplace=True)

# Save concatenated data in a single file for further processing - No data changes
output_file = (
    WORKING_DIR + "/Data/uci_heart_disease.original_processed.four_databases.tsv"
)
df.to_csv(output_file, sep="\t", index=False)
