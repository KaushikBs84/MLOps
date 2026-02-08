import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="KaushikBs/Tourism-Package", filename="best_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction App")
st.write("Tourism Package Prediction App is a private app for Visit with Us salespeople to predict if customer is likely to purchase a product based on their details")
st.write("Kindly enter the details to check whether the targeted customer is likely to buy the product.")

# Collect user input
TypeofContact = st.selectbox("The method by which the customer was contacted", ["Self Enquiry", "Company Invited"])
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("The city category based on development, population, and living standards", ["1", "2", "3"])
DurationOfPitch = st.number_input("Age (customer's age in years)", min_value=5, max_value=127, value=10)
Occupation = st.selectbox("Customer's occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Customer's gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Total number of people accompanying the customer on the trip",min_value=1, max_value=5, value=1)
NumberOfFollowups = st.number_input("Total number of follow-ups by the salesperson after the sales pitch",min_value=1, max_value=6, value=1)
ProductPitched = st.selectbox("The type of product pitched to the customer",["Basic","Deluxe","Standard","Super Deluxe","King"])
PreferredPropertyStar = st.selectbox("Preferred hotel rating by the customer",["3.0", "4.0", "5.0"])
MaritalStatus = st.selectbox("Customer's marital status", ["Married", "Single", "Divorced"])
NumberOfTrips = st.number_input("Average number of trips the customer takes annually",min_value=1, max_value=22, value=1)
Passport = st.selectbox("Has Credit Card?", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Score indicating the customer's satisfaction with the sales pitch",["1","2","3","4","5"])
OwnCar = st.selectbox("Is Active Member?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of children below age 5 accompanying the customer",min_value=0, max_value=3, value=0)
MonthlyIncome=st.number_input("Gross monthly income of the customer",min_value=1000, max_value=100000, value=10000)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': "1" if Passport == "Yes" else "0",
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': "1" if Passport == "Yes" else "0",
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Buy" if prediction == 1 else "Reject"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
