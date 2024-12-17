
# 2. Titanic Survival Prediction Using Streamlit and Kaggle API

## 1. Prompt: How can I fetch the Titanic dataset from Kaggle for analysis?

### Question:
How do I download the Titanic dataset from Kaggle using Python and ensure the data is ready for analysis?

### Solution:
We use the Kaggle API to authenticate and download the dataset. The data is stored locally in a specified directory and read into pandas DataFrames for further processing.

#### Code Snippet:
```python
import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

@st.cache_data
def download_kaggle_dataset():
    api = KaggleApi()
    try:
        api.authenticate() # Authenticate Kaggle API
    except Exception as e:
        st.error("Failed to authenticate Kaggle API. Please check your kaggle.json file.")
        return None, None

    dataset_path = './titanic'
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
```

---

## 2. Prompt: How can I create a Streamlit app to showcase Titanic data?

### Question:
How do I design a Streamlit app that displays and interacts with the Titanic dataset?

### Solution:
Streamlit's interface allows dynamic data visualization and user interaction. We set up the app to load the Titanic dataset and display its preview.

#### Code Snippet:
```python
import streamlit as st

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
else:
    st.error("Dataset download failed. Please check your Kaggle API settings and ensure you have joined the competition.")
```

---

## 3. Prompt: How do I integrate feature selection and model training?

### Question:
What methods can I use to perform feature selection and train a classification model for survival prediction?

### Solution:
This app integrates Recursive Feature Elimination (RFE), SelectKBest, and Optuna for feature selection. Logistic regression is used as the base classification model. The implementation is part of the data preparation section (to be extended).

---

## Example Outputs

### Streamlit App Preview
**App Interface**:
```text
Titanic Survival Prediction
This app uses RFE, SelectKBest, and Optuna for feature selection and logistic regression for classification.

Dataset downloaded and loaded successfully.
Train Data Preview:
   PassengerId  Survived  Pclass     Name     Sex  ...
Test Data Preview:
   PassengerId  Pclass     Name     Sex  Age  ...
```

---
![image](https://github.com/user-attachments/assets/1d85668e-0292-4f72-9ad2-8d433d419ecf)


## Notes
- Ensure the `kaggle.json` file is correctly set up in your system to authenticate the Kaggle API.
- Install all dependencies using:
  ```bash
  pip install pandas streamlit kaggle
  ```
- This app dynamically fetches the Titanic dataset and prepares it for further analysis.

