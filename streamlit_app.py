import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import requests
import os
from io import BytesIO
from sklearn.exceptions import InconsistentVersionWarning
import warnings

# Suppress the InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.title('🎈 Machine learning App')

st.write('Hello world!')

# File using for prediction
model_urls = ['https://raw.githubusercontent.com/HenryChritopher02/bace1/main/saved_model/model1.pkl',
              'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/saved_model/model2.pkl',
              'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/saved_model/model3.pkl',
              'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/saved_model/model4.pkl',
              'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/saved_model/model5.pkl']
train_files = [
    'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/QSAR-HB_M1_Train_6components.csv',
    'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/QSAR-HB_M2_Train_5components.csv',
    'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/QSAR-HB_M3_Train_3components.csv',
    'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/QSAR-HB_M4_Train_5components.csv',
    'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/QSAR-HB_M5_Train_5components.csv'
]

# Magic function
def predict_with_models(model_urls, data, train_files):

    # Initialize an empty DataFrame to store predictions
    result_df = pd.DataFrame({'SrNo': data['SrNo']})

    # Loop through each model URL and corresponding training file
    for i, (model_url, train_file_path) in enumerate(zip(model_urls, train_files)):
        # Download the model file from the URL
        response = requests.get(model_url)
        response.raise_for_status()  # Raise an error if the download fails

        # Load the model from the downloaded content
        model = joblib.load(BytesIO(response.content))

        # Read the training data
        train_data = pd.read_csv(train_file_path)
        
        # Extract feature columns (excluding 'SrNo' and 'pIC50_HB')
        cols_to_use = train_data.drop(columns=['SrNo', 'pIC50_HB']).columns.tolist()

        # Extract the corresponding features from the input data
        X = data[cols_to_use]

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_data[cols_to_use])
        X_scaled = scaler.transform(X)

        # Perform predictions
        y_pred = model.predict(X_scaled).ravel()

        # Add the predictions to the result DataFrame
        result_df[f'pIC50_Model_{i+1}'] = y_pred

    # Calculate the average prediction across all models
    result_df['Average_pIC50'] = result_df.iloc[:, 1:].mean(axis=1)

    return result_df
used_columns = ['VE3sign_D/Dt', 'P_VSA_s_4', 'SM14_AEA(dm)', 'CATS2D_09_AA', 'F07[O-O]', 'ATSC7dv', 'ATSC7i',	'VE1_B(p)', 'SsssNH+', 'P_VSA_LogP_2', 'TSRW10', 'P_VSA_ppp_P',	'VR3_Dzs',	'SssCH2	B06[N-N]',	'AATS8i']
with st.expander('**Input**'):
    option = st.radio("Choose an option", ("Upload CSV file", "Upload XLSX file"), index=None)
    if option == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            result = predict_with_models(model_urls, data, train_files)
            st.write('Descriptors for prediction')
            st.write(used_columns)
        else:
            st.error('Please upload a CSV file')

    elif option == "Upload XLSX file":
        uploaded_file = st.file_uploader("Choose a XLSX file", type='xlsx')
        if uploaded_file is not None:
            data = pd.read_excel(uploaded_file)
            result = predict_with_models(model_urls, data, train_files)
            st.write('Descriptors for prediction')
            st.write(used_columns)
        else:
            st.error('Please upload a XLSX file')
    else:
        st.error('Please choose your option')

with st.expander('**Prediction results**'):
    if 'result' in locals():
        st.write(result)
    else:
        st.error('No input data provided')
