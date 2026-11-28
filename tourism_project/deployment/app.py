import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tourism Package Predictor", page_icon="🧳")

# The model is committed to the repo by the GitHub Actions pipeline and sits next to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "tourism_package_v1_0.joblib")
THRESHOLD = 0.5   # same classification threshold used when the model was evaluated


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def model_problem(model):
    """Return a readable message if the loaded object is not a fitted pipeline, otherwise None."""
    if not hasattr(model, "named_steps"):
        return (f"The model file in the repository is a bare {type(model).__name__}, not the full "
                "preprocessing + XGBoost Pipeline. Re-run the GitHub Actions pipeline so it commits "
                "the fitted pipeline, then reboot this app.")
    try:
        model.named_steps["xgbclassifier"].get_booster()       # raises NotFittedError if unfitted
    except Exception:
        return ("The model file in the repository has not been fitted. Re-run the GitHub Actions "
                "pipeline so it commits the trained pipeline, then reboot this app.")
    return None


def ordered(options, preferred):
    """Show options in a natural order when we know it, otherwise keep the encoder's order."""
    return [p for p in preferred if p in options] + [o for o in options if o not in preferred]


model = load_model(MODEL_PATH)
problem = model_problem(model)
if problem:
    st.error(problem)
    st.stop()

# Allowed values for every categorical field come from the fitted encoder itself,
# so the form can never offer a label the model has not seen during training.
preprocessor = model.named_steps["columntransformer"]
cat_options = {}
for name, transformer, columns in preprocessor.transformers_:
    if name == "onehotencoder":
        for column, categories in zip(columns, transformer.categories_):
            cat_options[column] = [str(c) for c in categories]

yes_no = {1: "Yes", 0: "No"}

st.title("🧳 Wellness Tourism Package: Purchase Predictor")
st.write("Enter the customer details to predict whether the customer will purchase the package.")

left, right = st.columns(2)

with left:
    Age = st.number_input("Age", 18, 100, 35)
    TypeofContact = st.selectbox("Type of Contact", cat_options["TypeofContact"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox("Occupation", cat_options["Occupation"])
    Gender = st.selectbox("Gender", cat_options["Gender"])
    MaritalStatus = st.selectbox("Marital Status", cat_options["MaritalStatus"])
    Designation = st.selectbox(
        "Designation",
        ordered(cat_options["Designation"], ["Executive", "Manager", "Senior Manager", "AVP", "VP"]))
    MonthlyIncome = st.number_input("Monthly Income", 1000, 100000, 22000, step=500)
    Passport = st.selectbox("Passport", [1, 0], format_func=lambda v: yes_no[v])
    OwnCar = st.selectbox("Own Car", [1, 0], format_func=lambda v: yes_no[v])

with right:
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 3)
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (below 5)", 0, 5, 1)
    NumberOfTrips = st.number_input("Number of Trips per Year", 0, 25, 3)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    ProductPitched = st.selectbox(
        "Product Pitched",
        ordered(cat_options["ProductPitched"], ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]))
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 1, 130, 15)
    NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 4)
    PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 3)

# Save the inputs into a dataframe with exactly the columns (and order) used in training
input_data = pd.DataFrame([{
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
    "MonthlyIncome": MonthlyIncome,
}])[list(model.feature_names_in_)]

st.subheader("Input data sent to the model")
st.dataframe(input_data, hide_index=True)

if st.button("Predict Product Purchase"):
    probability = float(model.predict_proba(input_data)[0, 1])
    prediction = int(probability >= THRESHOLD)

    st.subheader("Prediction Result")
    if prediction == 1:
        #st.success(f"Product Taken: the customer is likely to purchase (probability {probability:.1%}).")
        st.success(f"Product Taken: the customer is likely to purchase")
    else:
        #st.warning(f"Product Not Taken: the customer is unlikely to purchase (probability {probability:.1%}).")
        st.warning(f"Product Not Taken: the customer is unlikely to purchase ")
