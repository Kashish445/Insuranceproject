import streamlit as st
import pickle
import pandas as pd

# Load the linear regression model from the pickle file
with open('pages/linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../decision_tree_pipeline_pre_1', 'rb') as f1:
    fraud_detection = pickle.load(f1)


# Streamlit web app
def main():
    st.title("Annual Premium Prediction")
    # Input fields for the five columns
    gender = st.selectbox('Gender', ['Male', 'Female'])
    vehicle_age = st.selectbox('Vehicle Age', ['< 1 Year', '1-2 Year', '> 2 Years'])
    driving_license = st.radio('Driving License', ['Yes', 'No'])
    previously_insured = st.radio('Previously Insured', ['Yes', 'No'])
    vehicle_age_years = st.number_input('Vehicle Age (in years)', min_value=0, max_value=100, step=1)
    pl = st.selectbox("Policy_Sales_Channel", ["Internal", "External"])
    vp = st.selectbox("Vehicle Price",
                      ["Less than 20000", "20000 - 29999", "30000 - 39999", "40000 - 49999", "50000 - 59999",
                       "60000 - 69999", "More than 70000"])
    vehicle_category = st.selectbox("Vehicle Category", ["Sport", "Utility", "Sedan"])
    ns = st.selectbox("Number of Suppliments", ["0", "1 to 2", "3 to 5", "More than 5"])
    fa = st.selectbox("Fault", ["Policy Holder", "Third Party"])
    ac = st.selectbox("Address Change", ["None", "1 year", "2 to 3 years", "4 to 8 years", "More than 8 years"])

    # Convert categorical inputs to numerical format
    gender_code = 1 if gender == 'Male' else 0
    vehicle_age_code = 0 if vehicle_age == '< 1 Year' else (1 if vehicle_age == '1-2 Year' else 2)
    driving_license_code = 1 if driving_license == 'Yes' else 0
    previously_insured_code = 1 if previously_insured == 'Yes' else 0

    if (pl == "Extrenal"):
        policy_sales_channel = 1
    else:
        policy_sales_channel = 0

    if (vp == "Less than 20000"):
        vehicle_price = 0
    elif (vp == "20000 - 29999"):
        vehicle_price = 1
    elif (vp == "30000 - 39999"):
        vehicle_price = 2
    elif (vp == "40000 - 49999"):
        vehicle_price = 3
    elif (vp == "50000 - 59999"):
        vehicle_price = 4
    elif (vp == "60000 - 69999"):
        vehicle_price = 5
    else:
        vehicle_price = 6

    if (vehicle_category == "Sport"):
        Sport = 1
        Sedan = 0
        Utility = 0
    elif (vehicle_category == "Sedan"):
        Sport = 0
        Sedan = 1
        Utility = 0
    else:
        Sport = 0
        Sedan = 0
        Utility = 1

    if (ns == "0"):
        number_suppliments = 0
    elif (ns == "1 to 2"):
        number_suppliments = 1
    elif (ns == "3 to 5"):
        number_suppliments = 2
    else:
        number_suppliments = 3

    if (fa == "Policy Holder"):
        fault = 1
    else:
        fault = 0

    if (ac == "None"):
        address_change = 0
    elif (ac == "1 year"):
        address_change = 1
    elif (ac == "2 to 3 years"):
        address_change = 2
    elif (ac == "4 to 8 years"):
        address_change = 3
    else:
        address_change = 4

    # Predict annual premium
    annual_premium = model.predict(
        [[gender_code, vehicle_age_code, driving_license_code, previously_insured_code, vehicle_age_years]])

    # Display predicted annual premium
    if st.button("Calculate Premium"):
        st.subheader("Predicted Annual Premium:")
        st.write(annual_premium[0])

    # Ask user for further action
    if st.button("Check for Fraud"):
        input_fraud = pd.DataFrame({
            'Sex': [gender_code],
            'premium': annual_premium,
            'AgeOfVehicle': [vehicle_age_code],
            'AgentType': [policy_sales_channel],
            'PastNumberOfClaims': [previously_insured_code],
            'VehiclePrice': [vehicle_price],
            # 'VehicleCategory': [vehicle_category],
            'DriverRating': [driving_license_code],
            'NumberOfSuppliments': [number_suppliments],
            'Fault': [fault],
            'AddressChange_Claim': [address_change],
            'VehicleCategory_Sedan': [Sedan],
            'VehicleCategory_Sport': [Sport],
            'VehicleCategory_Utility': [Utility],
        })
        fraud_detected = fraud_detection.predict_proba(input_fraud)
        if (fraud_detected[0][0] > fraud_detected[0][1]):
            st.write(f"There is a {(fraud_detected[0][0] * 100):.2f}% probability,the applicant might be Fraudulent!")
        else:
            st.write(f'The application seems genuine :{(fraud_detected[0][1] * 100):.2f}%')

    if st.button("Recheck Premium"):
        st.write("You can make changes to your inputs and recheck the premium.")


if __name__ == "__main__":
    main()
