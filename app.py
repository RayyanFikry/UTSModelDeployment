import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

st.title('Loan Prediction App')

with open('best_xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict_loan_status(features):
    print("Encoded features: ", features)

    try:
        encoded_features = [
            categorical_features['person_gender'][features[1]],  
            categorical_features['person_education'][features[2]],  
            categorical_features['person_home_ownership'][features[5]],  
            categorical_features['loan_intent'][features[7]]  
        ] + features[0:1] + features[3:5] + features[6:8] + features[9:]
    except KeyError as e:
        print(f"KeyError: The key {e}  - check if all categorical inputs are encoded correctly.")
        return None
        
    print("Encoded features: ", encoded_features)
    
    features_df = pd.DataFrame([encoded_features])
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    prediction = model.predict(scaled_features)
    return prediction[0]

person_age = st.number_input('Age of the Person', min_value=18, max_value=100, step=1)
person_gender = st.selectbox('Gender of the Person', ['Male', 'Female'])
person_education = st.selectbox('Education Level', ['High School', 'Bachelors', 'Masters', 'PhD'])
person_income = st.number_input('Annual Income', min_value=0, step=1000)
person_emp_exp = st.number_input('Years of Work Experience', min_value=0, step=1)
person_home_ownership = st.selectbox('Home Ownership Status', ['Own', 'Rent', 'Mortgage'])
loan_amnt = st.number_input('Loan Amount', min_value=0, step=500)
loan_intent = st.selectbox('Loan Intent', ['Personal', 'Business', 'Debt Consolidation'])
loan_int_rate = st.number_input('Loan Interest Rate (%)', min_value=0.0, step=0.1)
loan_percent_income = st.number_input('Loan as Percentage of Income', min_value=0.0, max_value=100.0, step=0.1)
cb_person_cred_hist_length = st.number_input('Credit History Length (in years)', min_value=0, step=1)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, step=1)
previous_loan_defaults_on_file = st.selectbox('Previous Loan Defaults', [0, 1])

input_features = [
    person_age,
    person_gender,
    person_education,
    person_income,
    person_emp_exp,
    person_home_ownership,
    loan_amnt,
    loan_intent,
    loan_int_rate,
    loan_percent_income,
    cb_person_cred_hist_length,
    credit_score,
    previous_loan_defaults_on_file
]

categorical_features = {
    'person_gender': {'Male': 1, 'Female': 0},
    'person_education': {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3},
    'person_home_ownership': {'Own': 0, 'Rent': 1, 'Mortgage': 2},
    'loan_intent': {'Personal': 0, 'Business': 1, 'Debt Consolidation': 2}
}

encoded_features = [
    categorical_features['person_gender'][input_features[1]], 
    categorical_features['person_education'][input_features[2]],  
    categorical_features['person_home_ownership'][input_features[5]], 
    categorical_features['loan_intent'][input_features[7]], 
] + input_features[0:1] + input_features[3:5] + input_features[6:8] + input_features[9:]

if st.button('Predict Loan Status'):
    prediction = predict_loan_status(input_features)
    
    if prediction is not None:
        if prediction == 1:
            st.success('Loan Approved')
        else:
            st.error('Loan Denied')
