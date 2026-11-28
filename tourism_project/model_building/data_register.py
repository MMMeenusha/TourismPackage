import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset straight from the repository data folder
df = pd.read_csv(RAW_PATH)

# Validate that every expected column is present before the dataset is "registered"
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact",
    "CityTier", "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Short summary printed in the workflow log
print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Missing values (total):", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("Product Taken distribution:")
print(df["ProdTaken"].value_counts())
