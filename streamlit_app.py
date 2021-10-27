import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import base64
import matplotlib.pyplot as plt
import matplotlib.dates as plt_date

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

st.write(dataset)
