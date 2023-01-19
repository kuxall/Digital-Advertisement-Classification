import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import streamlit as st

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


st.header("Digital Advertisment Classification")

df = pd.read_excel("data/ConcatenatedDigitalAdData.xlsx")
filename = '/home/kushal/Documents/projects/tf-idf-implementation/model/adv_model.sav'

st.text("The Classes for Classifications are:")

job_types = np.array(pd.DataFrame(df.JobType.unique()).values).flatten()
job_types_string = '\n'.join(job_types)

st.text(job_types_string)


input_text = st.text_area("Let's Input the text for Analysis", height=150, placeholder=None)


if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter the text for Classification. Thank You")
    else:
        df["TitleandDesc"] = df["title"] + df["Job_Description"]
        X = np.array(df["TitleandDesc"])
        y = np.array(df["JobType"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.score(X_test, y_test)
        st.success(f"The Accuracy Score is:  {round(result*100, 3)}%")    
        st.write("The Input Text is: ")
        st.caption(input_text)
        st.write("The Predicted Job Type is: ")
        prediction = loaded_model.predict([input_text])[0]
        st.success(prediction)
