import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.data_processor import preprocess_dataframe
from modules.predict import predict_sentiment
from modules.find_topics import extract_topics
from wordcloud import WordCloud

st.set_page_config(page_title='Streamlit App')

df = st.file_uploader(label="upload your csv file")
if df:
    df = pd.read_csv(df)
    df_cleaned = preprocess_dataframe(df, review_column='Full_Review')
    df_with_sentiment = predict_sentiment(df_cleaned, review_col='Full_Review')
    sentiment = df_with_sentiment["Sentiment"]
    reviews = df_with_sentiment["Full_Review"]
    cluster_indices, top_keywords,cluster_scores= extract_topics(df_cleaned['Full_Review'])
    df_cleaned['Topic_Cluster'] = cluster_indices
    df_cleaned['Top_Keywords'] = top_keywords
    
    
    st.markdown("<h3 style='font-weight: bold;'>Sentiment Overview</h3>", unsafe_allow_html=True)
    
    X= df_cleaned['Full_Review']
    y=df_cleaned['Sentiment']
    sentiment_counts = df_cleaned["Sentiment"].value_counts().sort_index()
    labels = ["Negative", "Neutral", "Positive"]
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=labels, autopct="%.2f%%")
    st.pyplot(fig)
    
    st.markdown("<h3 style='font-weight: bold;'>Topic Insights:</h3>", unsafe_allow_html=True)
    
    for cluster_id, keywords_scores in cluster_scores.items():
            keywords = [kw for kw, score in keywords_scores]
            scores = [score for kw, score in keywords_scores]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(keywords, scores, color='steelblue')
            ax.set_ylabel('TF-IDF Score', fontweight='bold')
            ax.set_xlabel('Keyword', fontweight='bold')
            ax.set_title(f'Top Keywords in Cluster {cluster_id}') 
            plt.tight_layout()
            st.pyplot(fig)
    
    st.markdown("<h3 style='font-weight: bold;'>Feedback Table</h3>", unsafe_allow_html=True)

    st.dataframe(
        df_cleaned[['Full_Review', 'Sentiment', 'Topic_Cluster', 'Top_Keywords']],
        column_config={
        "Full_Review": st.column_config.Column(width="large"),
        "Sentiment'": st.column_config.Column(width="small"),
        "Topic_Cluster": st.column_config.Column(width="small"),
        "Top_Keywords": st.column_config.Column(width="large"),
        },
        height=450
    )
    
    st.markdown("<h3 style='font-weight: bold;'>Word Clouds</h3>", unsafe_allow_html=True)

    def plot_sentiment_wordcloud(sentiment_label):
        filtered_reviews = df_with_sentiment[df_with_sentiment['Sentiment'] == sentiment_label]['Full_Review']
        text = " ".join(filtered_reviews.astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud for {sentiment_label} Reviews', fontweight='bold')
        st.pyplot(fig)
    
    plot_sentiment_wordcloud("Positive")
    plot_sentiment_wordcloud("Negative")