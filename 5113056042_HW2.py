import os
import pandas as pd
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi

# Step 0: Download Titanic Competition Dataset from Kaggle
@st.cache_data
def download_kaggle_dataset():
    api = KaggleApi()
    try:
        api.authenticate()  # 確保 Kaggle API 進行身份驗證
    except Exception as e:
        st.error("Failed to authenticate Kaggle API. Please check your kaggle.json file.")
        return None, None
    
    dataset_path = './titanic'  # 設置下載目錄
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        
    try:
        api.competition_download_files('titanic', path=dataset_path, unzip=True)
        train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
        test_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
        return train_data, test_data
    except Exception as e:
        st.error(f"Error downloading dataset: {e}")
        return None, None

# Streamlit setup
st.title("Titanic Survival Prediction")
st.write("This app uses RFE, SelectKBest, and Optuna for feature selection and logistic regression for classification.")

# Download and load the dataset
train_data, test_data = download_kaggle_dataset()
if train_data is not None and test_data is not None:
    st.write("Dataset downloaded and loaded successfully.")
    st.write("Train Data Preview:")
    st.write(train_data.head())
    st.write("Test Data Preview:")
    st.write(test_data.head())
    # Data preparation and model training code goes here...
else:
    st.error("Dataset download failed. Please check your Kaggle API settings and ensure you have joined the competition.")
