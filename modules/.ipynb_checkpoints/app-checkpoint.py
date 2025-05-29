import streamlit as st
import pandas as pd
import joblib
from modules.data_processor import preprocess_dataframe
from modules.predict import predict_sentiment

st.set_page_config(page_title='Streamlit App')
df = st.file_uploader(label="upload your csv file")
if df:
    df = pd.read_csv(df)
    df_cleaned = preprocess_dataframe(df, review_column='Full Review')
    df_with_sentiment = predict_sentiment(df_cleaned, review_col='Full Review')
    st.write("Data with Sentiment Predictions", df_with_sentiment[['Full Review', 'Sentiment']])
