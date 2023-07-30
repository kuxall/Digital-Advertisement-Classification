import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import streamlit as st

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def load_data(file_path):
    """Load the data from the given Excel file path."""
    return pd.read_excel(file_path)

def load_model(file_path):
    """Load the trained model from the given file path."""
    return pickle.load(open(file_path, 'rb'))

def preprocess_data(df):
    """Preprocess the data by concatenating 'title' and 'Job_Description'."""
    df["TitleandDesc"] = df["title"] + df["Job_Description"]
    return df

def display_job_types(df):
    """Display the classes for classifications."""
    job_types = np.array(pd.DataFrame(df.JobType.unique()).values).flatten()
    job_types_string = '\n'.join(job_types)
    st.text("The Classes for Classification are:")
    st.text(job_types_string)

def main(data_file, model_file):
    st.header("Digital Advertisement Classification")
    df = load_data(data_file)

    display_job_types(df)

    input_text = st.text_area("Let's Input the text for Analysis", height=150, placeholder=None)

    if st.button("Predict"):
        if input_text.strip() == "":
            st.warning("Please enter the text for Classification. Thank You")
        else:
            df = preprocess_data(df)
            X = np.array(df["TitleandDesc"])
            y = np.array(df["JobType"])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

            loaded_model = load_model(model_file)

            result = loaded_model.score(X_test, y_test)
            st.success(f"The Accuracy Score is: {round(result*100, 3)}%")

            st.write("The Input Text is: ")
            st.text(input_text)

            st.write("The Predicted Job Type is: ")
            prediction = loaded_model.predict([input_text])[0]
            st.success(prediction)

if __name__ == "__main__":
    # Provide the file paths for data and model
    data_file_path = "data/ConcatenatedDigitalAdData.xlsx"
    model_file_path = '/home/kushal/Documents/projects/tf-idf-implementation/model/adv_model.sav'
    main(data_file_path, model_file_path)
