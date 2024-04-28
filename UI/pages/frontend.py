import pickle
import streamlit as st
import pandas as pd

filename = 'pages/encoded_values.pkl'
with open(filename, 'rb') as file:
    encoded_values = pickle.load(file)

model_filename = 'pages/gb_model1.pkl'
with open(model_filename, 'rb') as file:
    gb_pipeline = pickle.load(file)

st.title('CLV Prediction')

insr_type_mapping = {
    1201: 'Private: 1201',
    1202: 'Commercial: 1202',
    1204: 'Motor Trade Road Risk : 1204'
}

# Input form for selected features
st.header('Enter Feature Values')
user_input = {}
user_input['SEX'] = st.selectbox('Sex', [0, 1])
user_input['CUSTOMER_LIFESPAN'] = st.number_input('Customer Lifespan')



# Create a selectbox using the mapped descriptions
user_input['INSR_TYPE'] = st.selectbox('Insurance Type', options=list(insr_type_mapping.values()), index=1)

# Reverse the mapping to get the numeric value based on the selected description
user_input['INSR_TYPE'] = next(key for key, value in insr_type_mapping.items() if value == user_input['INSR_TYPE'])

#selected_features=['SEX','CUSTOMER_LIFESPAN','INSR_TYPE','PREMIUM','USAGE','MAKE','TYPE_VEHICLE','Claim_Flag','Sales_Channel','Age']


#user_input['INSURED_VALUE'] = st.number_input('Insured Value')
user_input['PREMIUM'] = st.number_input('Premium')
user_input['Age'] = st.number_input('Age')
user_input['USAGE']= st.selectbox('Usage', list(encoded_values['USAGE'].keys()))
user_input['MAKE'] = st.selectbox('Make', list(encoded_values['MAKE'].keys()))
user_input['TYPE_VEHICLE'] = st.selectbox('Type of Vehicle', list(encoded_values['TYPE_VEHICLE'].keys()))
user_input['Claim_Flag'] = st.selectbox('Claim Flag', [0, 1])
user_input['Sales_Channel'] = "internal"

compute_clv = st.button('Compute CLV')

if compute_clv:
    
    user_data = pd.DataFrame([user_input])

    # Predict CLV
    predictions = gb_pipeline.predict(user_data)

    # Display predicted CLV
    st.header('Predicted CLV')
    st.write(f'The predicted Customer Lifetime Value (CLV) is: {predictions}')
