import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lime
from lime.lime_tabular import LimeTabularExplainer

# Load the linear regression model from the pickle file
with open('linear_regr_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('pages/decision_tree_pipeline_pre_1.2.2', 'rb') as f1:
    fraud_detection = pickle.load(f1)
filename = 'pages\encoded_values.pkl'
with open(filename, 'rb') as file:
    encoded_values = pickle.load(file)
    
model_filename = 'pages\gb_model1.pkl'
with open(model_filename, 'rb') as file:
    gbpipeline = pickle.load(file)

feature_names_mapping = {
    'premium': 'Premium',
    'AgeOfVehicle': 'Age of Vehicle',
    'AgentType': 'Agent Type',
    'PastNumberOfClaims': 'Past Number of Claims',
    'VehiclePrice': 'Vehicle Price',
    'DriverRating': 'Driver Rating',
    'NumberOfSuppliments': 'Number of Suppliments',
    'Fault': 'Fault',
    'AddressChange_Claim': 'Address Change (Claim)',
    'vehicle_cat_n': 'Vehicle Category',
}

insr_type_mapping = {
            1201: 'Private: 1201',
            1202: 'Commercial: 1202',
            1204: 'Motor Trade Road Risk : 1204'
        }

def convert_except_last(string):
    words = string.split()
    result = ' '.join(words[:-2])
    return result

# Streamlit web app
def main():
    gender = 0
    driving_License = 0
    previously_insured = 0
    vehicle_age = 0
    policy_sales_channel = 0
    vehicle_price = 0
    vehicle_category_m =0
    number_suppliments = 0
    fault = 0
    address_change = 0


    st.title("Annual Premium Prediction")
    # Input fields for the five columns
    gd = st.selectbox('Gender', ['Male', 'Female'])
    dl = st.radio('Driving License', ['Yes', 'No'])
    pi = st.radio('Previously Insured', ['Yes', 'No'])
    va = st.selectbox("Vehicle Age",["0 years","1 year","2 years","3 years","4 years","5 years","6 years","7 years","More than 7 years"])

    pl = st.selectbox("Policy_Sales_Channel", ["Internal", "External"])
    vp = st.selectbox("Vehicle Price",
                      ["Less than 20000", "20000 - 29999", "30000 - 39999", "40000 - 49999", "50000 - 59999",
                       "60000 - 69999", "More than 70000"])
    vehicle_category = st.selectbox("Vehicle Category", ["Sport", "Utility", "Sedan"])
    ns = st.selectbox("Number of Suppliments", ["0", "1 to 2", "3 to 5", "More than 5"])
    fa = st.selectbox("Fault", ["Policy Holder", "Third Party"])
    ac = st.selectbox("Address Change", ["None", "1 year", "2 to 3 years", "4 to 8 years", "More than 8 years"])


    # Convert categorical inputs to numerical format
    if(gd=='Male'):
        gender=1
    else:
        gender=0

    if(dl=='Yes'):
        driving_License=1
    else:
        driving_License=0

    if (pi=='Yes'):
        previously_insured=1
    else:
        previously_insured=0

    if (va == "0 years"):
        vehicle_age_code = 0
    elif (va == "1 year" or va == "2 years"):
        vehicle_age_code = 1
    else:
        vehicle_age_code = 2

    if(va=="0 years"):
        vehicle_age_c=0
    elif(va == "1 year"):
        vehicle_age_c=1
    elif(va == "2 years"):
        vehicle_age_c = 2
    elif (va == "3 years"):
        vehicle_age_c = 3
    elif (va == "4 years"):
        vehicle_age_c = 4
    elif (va == "5 years"):
        vehicle_age_c = 5
    elif (va == "6 years"):
        vehicle_age_c = 6
    elif (va == "7 years"):
        vehicle_age_c = 7
    else:
        vehicle_age_c=8



    if (pl == "External"):
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
        vehicle_category_m=1
    elif (vehicle_category == "Sedan"):
        vehicle_category_m=0
    else:
        vehicle_category_m=2

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
    annual_premium = model.predict([[gender, driving_License, previously_insured, vehicle_age_code, policy_sales_channel]])

    user_input = {}
    user_input['SEX'] = gender
    user_input['PREMIUM'] = annual_premium
    user_input['Age'] = vehicle_age_code

    user_input['CUSTOMER_LIFESPAN'] = st.slider('Customer Lifespan in days', min_value=0, max_value=365, value=180)

    # Create a selectbox using the mapped descriptions
    user_input['INSR_TYPE'] = st.selectbox('Insurance Type', options=list(insr_type_mapping.values()), index=1)

    # Reverse the mapping to get the numeric value based on the selected description
    user_input['INSR_TYPE'] = next(
        key for key, value in insr_type_mapping.items() if value == user_input['INSR_TYPE'])

    insured_value = st.slider('Insured Value', min_value=0, max_value=100000, value=0)

    user_input['USAGE'] = st.selectbox('Usage', list(encoded_values['USAGE'].keys()))
    user_input['MAKE'] = st.selectbox('Make', list(encoded_values['MAKE'].keys()))
    user_input['TYPE_VEHICLE'] = st.selectbox('Type of Vehicle', list(encoded_values['TYPE_VEHICLE'].keys()))
    cf = st.selectbox('Do you wanna Claim?', ['No', 'Yes'])
    user_input['Claim_Flag'] = 1 if cf == 'Yes' else 0
    user_input['Sales_Channel'] = "internal"
    user_data = pd.DataFrame([user_input])
    predictions = gbpipeline.predict(user_data)
    lifespan = user_input['CUSTOMER_LIFESPAN']
    unrealized_value = annual_premium - (insured_value * lifespan)
    unrealized_status = "Unrealized Profit" if unrealized_value > 0 else "Unrealized Loss"

    # Display predicted annual premium
    if st.button("Calculate Premium"):
        st.subheader("Predicted Annual Premium:")
        st.write(annual_premium[0])

    # Ask user for further action
    if st.button("Check for Fraud"):
        input_n = pd.DataFrame({
            'premium': annual_premium,
            'AgeOfVehicle': [vehicle_age_code],
            'AgentType': [policy_sales_channel],
            'PastNumberOfClaims': [previously_insured],
            'VehiclePrice': [vehicle_price],
            # 'VehicleCategory': [vehicle_category],
            'DriverRating': [driving_License],
            'NumberOfSuppliments': [number_suppliments],
            'Fault': [fault],
            'AddressChange_Claim': [address_change],
            'vehicle_cat_n':[vehicle_category_m]})
        print(input_n)
        fraud_detected = fraud_detection.predict_proba(input_n)
        if (fraud_detected[0][0]>fraud_detected[0][1]):
            st.write(f"There is a {(fraud_detected[0][0]*100):.2f}% probability,the applicant might be Fraudulent!")
            explainer = lime.lime_tabular.LimeTabularExplainer(input_n.values,
                                                               feature_names=list(map(lambda x: feature_names_mapping[x], input_n.columns)),
                                                               class_names=['Fraudulent', 'Not Fraudulent'],
                                                               discretize_continuous=True)
            exp = explainer.explain_instance(input_n.iloc[0], fraud_detection.predict_proba, num_features=3)
            st.subheader("Top 3 Features Contributing to Fraudulence:")
            for feature, weight in exp.as_list():
                feature_name = convert_except_last(feature)
                st.write(f"- {feature_name}")
        else:
            st.write(f'The application seems genuine :{(fraud_detected[0][1]*100):.2f}%')
            if(fraud_detected[0][1]>0.5):
                explainer = lime.lime_tabular.LimeTabularExplainer(input_n.values,
                                                                   feature_names=list(
                                                                       map(lambda x: feature_names_mapping[x],
                                                                           input_n.columns)),
                                                                   class_names=['Fraudulent', 'Not Fraudulent'],
                                                                   discretize_continuous=True)
                exp = explainer.explain_instance(input_n.iloc[0], fraud_detection.predict_proba, num_features=3)
                st.subheader("Top 3 Features Contributing to Fraudulence:")
                for feature, weight in exp.as_list():
                    feature_name = convert_except_last(feature)
                    st.write(f"- {feature_name}")

    if st.button("Recheck Premium"):
        st.write("You can make changes to your inputs and recheck the premium.")
    if st.button("Check CLV"):
        st.title('CLV Prediction')
        st.write(f'The predicted Customer Lifetime Value (CLV) is: {predictions}')
        st.write(f'The calculated Unrealized Value is: {unrealized_value}')
        st.write(f'The status is: {unrealized_status}')

if __name__ == "__main__":
    main()