import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def extract_topics(reviews):
    global order_centroids, terms ,vectors
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=100,
        max_df=0.8,
        min_df=2,
        ngram_range=(1, 3),
        stop_words='english'
    )
    vectors = vectorizer.fit_transform(reviews)
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()

    all_keywords = []

    for review in denselist:
        x=0
        keywords = []
        for word in review:
            if word>0:
                keywords.append(feature_names[x])
            x=x+1
        all_keywords.append(keywords)
    model = KMeans(n_clusters=5,init="k-means++",max_iter=100,n_init=10)
    model.fit(vectors)
    kmean_indices = model.fit_predict(vectors)

    order_centroids = model.cluster_centers_.argsort()[:,::-1]
    terms = vectorizer.get_feature_names_out()

    cluster_keywords = {
        i: [terms[ind] for ind in order_centroids[i, :5]]
        for i in range(5)
    }
    review_keywords = [cluster_keywords[label] for label in kmean_indices]
    cluster_scores = {}

    for cluster_id in range(model.n_clusters):
        keywords = [
            (terms[ind], round(model.cluster_centers_[cluster_id][ind], 4))
            for ind in order_centroids[cluster_id, :5]
            ]
        cluster_scores[cluster_id] = keywords

    return kmean_indices, review_keywords,cluster_scores