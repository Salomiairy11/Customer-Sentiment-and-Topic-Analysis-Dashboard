import streamlit as st
import pandas as pd
st.set_page_config(page_title='Streamlit App')
df = st.file_uploader(label="upload your csv file")
if df:
    df = pd.read_csv(df)
    st.write(df.head())