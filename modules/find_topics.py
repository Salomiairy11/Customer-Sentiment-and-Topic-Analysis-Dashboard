import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


processed_data = pd.read_pickle("../models/data_processed.pkl")
df_cleaned = pd.read_pickle("../models/data_cleaned.pkl")
reviews = df_cleaned["Full Review"]
sentiment = df_cleaned["Sentiment"]

def extract_topics(reviews, n_clusters=5):
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
        for i in range(n_clusters)
    }
    review_keywords = [cluster_keywords[label] for label in kmean_indices]

    return kmean_indices, review_keywords

cluster_indices, top_keywords = extract_topics(df_cleaned['Full Review'])

df_cleaned['Topic Cluster'] = cluster_indices
df_cleaned['Top Keywords'] = top_keywords

with open("../data/trc_result.txt", "w", encoding="utf-8") as f:
    for i in range(5): 
        f.write(f"Cluster{i}\n")
        for ind in order_centroids[i, :5]:
            f.write(f" {terms[ind]}\n")
        f.write("\n\n")

pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vectors.toarray())

colors = ["r","b","c","y","m"]

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

fig,ax = plt.subplots(figsize=(50,50))

ax.scatter(x_axis,y_axis,c=[colors[d] for d in cluster_indices])

plt.savefig("trc1.png")