################################################################################################
# Author: Joan Ponsa
# Created on: 28/10/2023
# Last modified: 29/10/2023
#
# Takes the original UCI heart disease dataset and does any data transformation prior ML
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

# Load original dataset
# fmt: off
input_file = WORKING_DIR + "data/uci_heart_disease.original_processed.four_databases.tsv"
# fmt: on
df = pd.read_csv(input_file, sep="\t")

# Create a binary version of variable num
df["num_01"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

# Data correction
# cholesterol values > 400 are outliers, replacing them with 401
df["chol"].mask((df["chol"].notna() & df["chol"] > 400), 401, inplace=True)
# values of 0 cholesterol make no sense. I will assume any 0 value is a missing value.
df["chol"].replace(0, np.nan, inplace=True)

# values of 0 trestbps make no sense. I will assume any 0 value is a missing value.
df["trestbps"].replace(0, np.nan, inplace=True)
# I believe negative "oldpeak" make no sense. Replacing any negative value with NaN
df["oldpeak"].mask(df["oldpeak"] < 0, np.nan, inplace=True)

# Recode variables
# cp: 1 = typica angina, 2= atypical angina, 3=non-aginal pain, 4= asymptomatic
df["cp_1"] = 0
df.loc[df["cp"] == 1, "cp_1"] = 1
df["cp_2"] = 0
df.loc[df["cp"] == 2, "cp_2"] = 1
df["cp_3"] = 0
df.loc[df["cp"] == 3, "cp_3"] = 1

# restecg 0 = normal, 1 = abnormality, 2 = left ventricular hypertrophy
df["restecg_1"] = 0
df.loc[df["restecg"] == 1, "restecg_1"] = 1
df["restecg_2"] = 0
df.loc[df["restecg"] == 2, "restecg_2"] = 1

# slope: 1 = upsloping, 2 =  flat, 3 = downslopling
df["slope"].replace({1: 1, 2: 0, 3: -1}, inplace=True)

# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
df["thal_6"] = 0
df.loc[df["thal"] == 6, "thal_6"] = 1
df["thal_7"] = 0
df.loc[df["thal"] == 7, "thal_7"] = 1

# TODO: Imputation of missing values

# Remove the VA data center due to poor data quality
df = df[df["data_center"] != "va"]


# Feature manually selected based on the low number of missing values
feature_subset = [
    "age",
    "sex",
    "cp_1",
    "cp_2",
    "cp_3",
    "trestbps",
    "restecg_1",
    "restecg_2",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "num",
    "num_01",
]
# Trim feature to only the selected subset
df = df[feature_subset]
# remove any values with missing records
df.dropna(inplace=True)

# Save preprocessed data
output_file = WORKING_DIR + "data/uci_heart_disease.processed.tsv"
df.to_csv(output_file, sep="\t", index=False)
