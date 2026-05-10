import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model and preprocessing objects
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")

# Collect user input
geography = st.selectbox("Geography", one_hot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100)
balance = st.number_input("Balance", min_value=0.0)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
credit_score = st.slider("Credit Score", min_value=300, max_value=850)
tenure = st.slider("Tenure (years)", min_value=0, max_value=10)

if st.button("Predict Churn"):

    # Prepare input data
    input_data = {
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': 1 if has_cr_card == "Yes" else 0,
        'IsActiveMember': 1 if is_active_member == "Yes" else 0,
        'EstimatedSalary': estimated_salary,
        'CreditScore': credit_score,
        'Tenure': tenure
    }

    ## One hot encode Geography
    geo_encoded = one_hot_encoder.transform(
        [[input_data['Geography']]]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_encoder.get_feature_names_out(['Geography'])
    )

    ## Combine encoded geography with input data
    input_data = pd.concat(
        [pd.DataFrame([input_data]), geo_encoded_df],
        axis=1
    )

    ## Reorder columns
    input_data = input_data.reindex(columns=[
        'CreditScore',
        'Geography',
        'Gender',
        'Age',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'HasCrCard',
        'IsActiveMember',
        'EstimatedSalary',
        'Geography_France',
        'Geography_Germany',
        'Geography_Spain'
    ])

    ## Encode Gender
    input_data['Gender'] = label_encoder_gender.transform(
        input_data['Gender']
    )

    ## Drop original Geography column
    input_data = input_data.drop(columns=['Geography'])

    ## Scale input data
    input_data_scaled = scaler.transform(input_data)

    ## Predict churn
    churn_probability = model.predict(input_data_scaled)

    st.write(
        f"Predicted probability of churn: {churn_probability[0][0]:.4f}"
    )

    if churn_probability[0][0] > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")