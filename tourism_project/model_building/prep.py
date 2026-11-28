import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"

# 1) Load the dataset directly from the repository data folder
df = pd.read_csv(RAW_PATH)
print("Raw shape:", df.shape)

# 2) Clean the data
# 2a) strip stray whitespace from every text column
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].str.strip()

# 2b) fix inconsistent labels (same meaning, different spelling)
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# 2c) drop rows with missing values (the raw file has none, this is a safety net)
before = len(df)
df = df.dropna()
print("Rows dropped because of missing values:", before - len(df))

# 2d) remove unnecessary columns: CustomerID is only an identifier and carries no predictive signal
df = df.drop(columns=["CustomerID"])

# NOTE: no manual encoding here. Categorical columns stay as raw text and are one-hot encoded
# INSIDE the training pipeline, so training and the Streamlit app use exactly the same representation.

# 3) Split into train and test sets (stratified because the target is imbalanced)
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Save the splits locally; the workflow uploads them as an artifact for the next job
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Xtrain:", Xtrain.shape, "| Xtest:", Xtest.shape)
print("Share of buyers in train/test:", round(ytrain.mean(), 3), "/", round(ytest.mean(), 3))
