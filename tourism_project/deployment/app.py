
import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# Load the model
model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tourism_package_v1_0.joblib"
)

st.title("Tourism Package App")

st.write("""
This application predicts whether a customer will purchase a tourism package.
""")

# Check model file
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Load model
model = joblib.load(model_path)

# Verify model type
if not isinstance(model, Pipeline):
    st.error(
        f"Incorrect model loaded: {type(model)}. "
        "Expected a fitted Scikit-learn Pipeline."
    )
    st.stop()

# Verify XGBoost is fitted
try:
    model.named_steps["xgbclassifier"].get_booster()
except Exception as e:
    st.error(f"The loaded XGBoost model is not fitted: {e}")
    st.stop()

st.success("Fitted model loaded successfully!")

# Customer details
CustomerID = st.text_input("CustomerID", "12345")
Age = st.number_input("Age", 0, 100, 30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch", 0, 100, 30)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

Gender = st.selectbox("Gender", ["Male", "Female"])

NumberOfPersonVisiting = st.number_input(
    "Number of Person Visiting", 0, 10, 2
)

NumberOfFollowups = st.number_input("Number of Followups", 0, 10, 2)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.number_input(
    "Preferred Property Star", 1, 5, 3
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced"]
)

NumberOfTrips = st.number_input("Number of Trips", 0, 10, 2)

Passport = st.selectbox("Passport", [1, 0])

PitchSatisfactionScore = st.number_input(
    "Pitch Satisfaction Score", 1, 5, 3
)

OwnCar = st.selectbox("Own Car", [1, 0])

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting", 0, 10, 2
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income", 0, 100000, 50000
)

# Prepare input data
raw_input = {
    "CustomerID": int(CustomerID) if CustomerID.isdigit() else 0,
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}

input_data = pd.DataFrame([raw_input])

# Maintain exactly the same features used during training
expected_features = [
    "CustomerID", "Age", "CityTier", "DurationOfPitch",
    "NumberOfPersonVisiting", "NumberOfFollowups", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "PreferredPropertyStar",
    "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
    "TypeofContact", "Occupation", "Gender",
    "MaritalStatus", "ProductPitched", "Designation"
]

input_data = input_data[expected_features]

# Prediction
if st.button("Predict Product Purchase"):

    try:
        prediction = model.predict(input_data)[0]

        result = (
            "Product Taken"
            if prediction == 1
            else "Product Not Taken"
        )

        st.subheader("Prediction Result:")
        st.success(f"The model predicts: **{result}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
