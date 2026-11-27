import os
import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "tourism_package_v1_0.joblib")
model = joblib.load(model_path)

st.title("Tourism Package App")
st.write("""
This application predicts the customer will purchase a package or not.
""")

CustomerID = st.text_input("CustomerID", "12345")
Age = st.number_input("Age", 0, 100, 30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch", 0, 100, 30)
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Person Visiting", 0, 10, 2)
NumberOfFollowups = st.number_input("Number of Followups", 0, 10, 2)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.number_input("Preferred Property Star", 1, 5, 3)
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips", 0, 10, 2)
Passport = st.selectbox("Passport", [1, 0])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Own Car", [1, 0])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", 0, 10, 2)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", 0, 100000, 50000)

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

# Automatically match exact column names and order used during training
if hasattr(model, "feature_names_in_"):
    input_data = input_data[list(model.feature_names_in_)]
else:
    input_data = input_data.drop(columns=["CustomerID"])

if st.button("Predict Product Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "Product Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**") 
