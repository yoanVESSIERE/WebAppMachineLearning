import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import base64
import matplotlib.pyplot as plt
import matplotlib.dates as plt_date
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from datetime import datetime as dt

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="US Health Insurance Cost", page_icon="💚", layout='centered', initial_sidebar_state="collapsed")

dataset = pd.read_csv('data/insurance.csv')

fd = open("./image/healing.gif", "rb")
contents = fd.read()
data_url = base64.b64encode(contents).decode("utf-8")
fd.close()

st.markdown(
    f'<p align="center"><img width=200 src="data:image/gif;base64,{data_url}" alt="cat gif"></p>',
    unsafe_allow_html=True,
)

st.title("The US Health Insurance Prediction")

col1, col2 = st.columns(2)

with col1:
    st.header('Informations')
    st.write("The information part will give you some information about this webapp !")
    st.write("Here, you can have a prediction on how much you'll pay in health insurance in the US depending on different factor.")
    st.write('For that, you just have to enter some information in the "Prediction" part, click on the "Let\'s go !" button and you\'ll see aproximatively how much you would have paid')
    st.markdown('**Info:** The BMI is the Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9')

with col2:
    st.header('Prediction')

    sex = st.selectbox(
        'What is your gender ?',
        ('Male', 'Female')
    )

    age = st.number_input('How old are you ?', min_value=0, value=20)

    smoker = st.selectbox(
        'Are you a smoker ?',
        ('Yes', 'No')
    )

    bmi = st.number_input('Insert your BMI', min_value=1.0, value=30.66)

    child = st.number_input('How many childs have you ? (Zero if not)', min_value=0, value=0)

    button = st.button('Let\'s go !')

if (button):
    smoker_dict = {
        "no": 0,
        "yes": 1
        }
    sex_dict = {
        "female": 0,
        "male": 1
        }
    dataset["smoker"] = dataset["smoker"].apply(lambda x: smoker_dict[x] if x in smoker_dict else 0)
    dataset['sex'] = dataset['sex'].apply(lambda x: sex_dict[x] if x in sex_dict else 0)

    dataset.drop('region', axis=1, inplace=True)

    X = dataset[['smoker', 'bmi', 'age', 'sex', 'children']]
    Y = dataset['charges']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

    reg = LinearRegression().fit(X_train, Y_train)

    Y_predict = reg.predict(X_test)
    prediction = reg.predict([[smoker_dict[smoker.lower()], bmi, age, sex_dict[sex.lower()], child]])[0]

    y_true = Y_test
    y_pred = Y_predict
    score = r2_score(y_true, y_pred)

    st.markdown("The Machine Learning predict a health bill of **" + str(format(prediction, '.2f')) + " $** with a R2 score of **" + str(score) + "** /1")
