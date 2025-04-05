import streamlit as st
import pickle
import pandas as pd

with open("df.pkl","rb") as a:
  df = pd.DataFrame(a)

st.header("House Price Prediction")
st.selectbox("Select Gender",["Male","female","Others"])
st.selectbox("Select Sector",df["sector"].unique().to_list())
