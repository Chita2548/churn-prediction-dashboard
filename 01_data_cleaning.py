import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw\WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.drop(columns=["customerID"], inplace=True)


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

cat_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df[num_cols] = scaler.fit_transform(df[num_cols])

df.to_csv("data/processed/cleaned_data.csv", index=False)

print("Data cleaning completed")