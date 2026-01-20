
import os
import subprocess
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATASET = "blastchar/telco-customer-churn"
OUTPUT_DIR = "data/raw"

KAGGLE_EXE = r"C:\Users\PC\Desktop\Mini project 1\venv\Scripts\kaggle.exe"
KAGGLE_CONFIG_DIR = r"C:\Users\PC\Desktop\Mini project 1\.kaggle"

os.makedirs(OUTPUT_DIR, exist_ok=True)

env = os.environ.copy()
env["KAGGLE_CONFIG_DIR"] = KAGGLE_CONFIG_DIR

subprocess.run(
    [
        KAGGLE_EXE,
        "datasets",
        "download",
        DATASET,
        "-p",
        OUTPUT_DIR,
        "--unzip"
    ],
    check=True,
    env=env
)

print("✅ Dataset downloaded successfully")

