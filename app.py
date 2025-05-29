import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from modules.data_processor import preprocess_dataframe
from modules.predict import predict_sentiment
from modules.find_topics import extract_topics

st.set_page_config(page_title='Streamlit App')
df = st.file_uploader(label="upload your csv file")
if df:
    df = pd.read_csv(df)
    df_cleaned = preprocess_dataframe(df, review_column='Full Review')
    df_with_sentiment = predict_sentiment(df_cleaned, review_col='Full Review')
    sentiment = df_with_sentiment["Sentiment"]
    reviews = df_with_sentiment["Full Review"]
    
    cluster_indices, top_keywords = extract_topics(df_cleaned['Full Review'])

    df_cleaned['Topic Cluster'] = cluster_indices
    df_cleaned['Top Keywords'] = top_keywords

    st.write("Classified & Clustered Reviews")
    st.dataframe(df_cleaned[['Full Review', 'Sentiment', 'Topic Cluster', 'Top Keywords']])
    #st.write("Data with Sentiment Predictions", df_with_sentiment[['Full Review', 'Sentiment']])
