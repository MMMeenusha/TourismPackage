import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("tourism_project/data/tourism.csv")
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

# Define the order
product_order = ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King']
# Initialize encoder
encoder = OrdinalEncoder(categories=[product_order])
# Apply encoding
df[['ProductPitched']] = encoder.fit_transform(df[['ProductPitched']])
# Check the result
df['ProductPitched'].value_counts()

# Define the order
designation_order = ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP']
# Initialize encoder
encoder = OrdinalEncoder(categories=[designation_order])
# Apply encoding
df[['Designation']] = encoder.fit_transform(df[['Designation']])
# Check the result
df['Designation'].value_counts()

# NOTE: 'Type' is intentionally left as raw strings (H/L/M).
# The training pipeline one-hot-encodes it, and the Streamlit app also sends
# raw H/L/M values. Encoding it here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Type values kept as:", sorted(X["Type"].unique()))
