import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.data_processor import preprocess_dataframe
from modules.predict import predict_sentiment
from modules.find_topics import extract_topics
from wordcloud import WordCloud

st.set_page_config(page_title='Customer Feedback Dashboard', layout='wide')

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
df = st.file_uploader(label="Upload your CSV file")
if df:
    df = pd.read_csv(df)
    df_cleaned = preprocess_dataframe(df, review_column='Full_Review')
    df_with_sentiment = predict_sentiment(df_cleaned, review_col='Full_Review')
    cluster_indices, top_keywords, cluster_scores = extract_topics(df_cleaned['Full_Review'])
    
    df_cleaned['Sentiment'] = df_with_sentiment['Sentiment']
    df_cleaned['Topic_Cluster'] = cluster_indices
    df_cleaned['Top_Keywords'] = top_keywords

    st.markdown("<h1 class='dashboard-title'>Customer Feedback Analysis Dashboard</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    sentiment_counts = df_cleaned["Sentiment"].value_counts().sort_index()
    labels = ["Negative", "Neutral", "Positive"]
    with col1:
        st.markdown("<h3>Sentiment Overview</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.pie(sentiment_counts, labels=labels, autopct="%.2f%%", colors=['#ef4444','#fbbf24','#34d399'])
        st.pyplot(fig)

    with col2:
        st.markdown("<h3>Positive Reviews</h3>", unsafe_allow_html=True)
        pos_text = " ".join(df_with_sentiment[df_with_sentiment['Sentiment']=='Positive']['Full_Review'].astype(str))
        pos_wc = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(pos_text)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.imshow(pos_wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    with col3:
        st.markdown("<h3>Negative Reviews</h3>", unsafe_allow_html=True)
        neg_text = " ".join(df_with_sentiment[df_with_sentiment['Sentiment']=='Negative']['Full_Review'].astype(str))
        neg_wc = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(neg_text)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.imshow(neg_wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    st.markdown("<h3>Topic Insights</h3>", unsafe_allow_html=True)
    num_cols = min(len(cluster_scores), 3)
    topic_cols = st.columns(num_cols)
    bar_colors = ['#3b82f6','#06b6d4','#2563eb']

    for i, (cluster_id, keywords_scores) in enumerate(cluster_scores.items()):
        keywords = [kw for kw, score in keywords_scores]
        scores = [score for kw, score in keywords_scores]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(keywords, scores, color=bar_colors[i % len(bar_colors)])
        ax.set_ylabel('TF-IDF Score', fontweight='bold')
        ax.set_xlabel('Keyword', fontweight='bold')
        ax.set_title(f'Top Keywords in Cluster {cluster_id}', fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        topic_cols[i % num_cols].pyplot(fig) 

    st.markdown("<h3>Feedback Table</h3>", unsafe_allow_html=True)
    st.dataframe(
        df_cleaned[['Full_Review', 'Sentiment', 'Topic_Cluster', 'Top_Keywords']],
        column_config={
            "Full_Review": st.column_config.Column(width="large"),
            "Sentiment": st.column_config.Column(width="small"),
            "Topic_Cluster": st.column_config.Column(width="small"),
            "Top_Keywords": st.column_config.Column(width="large"),
        },
        height=450
    )
