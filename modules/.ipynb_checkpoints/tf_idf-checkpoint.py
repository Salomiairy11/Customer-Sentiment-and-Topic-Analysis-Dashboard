import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
from nltk.corpus import stopwords
import json
import glob
import re
from data_processor import preprocess_dataframe

processed_data = pd.read_pickle("../models/data_processed.pkl")
df_cleaned = df_cleaned = pd.read_pickle("../models/data_cleaned.pkl")
reviews = df_cleaned["Full Review"]
sentiment = processed_data["Sentiment"]

vectorizer = TfidfVectorizer(
                             lowercase=True,
                             max_features=100,
                             max_df=0.8,
                             min_df=5,
                             ngram_range = (1,3),
                             stop_words = 'english'
                             )

vectors = vectorizer.fit_transform(df_cleaned["Full Review"])
feature_names = vectorizer.get_feature_names_out()
dense = vectors.todense()
denselist = dense.tolist()

all_keywords = []

for reviews in denselist:
    x=0
    keywords = []
    for word in reviews:
        if word>0:
            keywords.append(feature_names[x])
        x=x+1
    all_keywords.append(keywords)

print(reviews[1],all_keywords[1])