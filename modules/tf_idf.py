import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

processed_data = pd.read_pickle("../models/data_processed.pkl")
df_cleaned = pd.read_pickle("../models/data_cleaned.pkl")
reviews = df_cleaned["Full_Review"]
sentiment = df_cleaned["Sentiment"]

vectorizer = TfidfVectorizer(lowercase=True,
                            max_features=100,
                            max_df=0.8,
                            min_df=5,
                            ngram_range = (1,3),
                            stop_words = 'english')

vectors = vectorizer.fit_transform(df_cleaned["Full_Review"])
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

true_k = 5

model = KMeans(n_clusters=true_k,init="k-means++",max_iter=100,n_init=1)
model.fit(vectors)

order_centroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names_out()

with open("../data/trc_results.txt","w",encoding="utf-8") as f:
    for i in range(true_k):
        f.write(f"Cluster{i}")
        f.write("\n")
        for ind in order_centroids[i,:5]:
            f.write( ' %s' % terms[ind],)
            f.write("\n")
        f.write("\n")
        f.write("\n")

kmean_indices = model.fit_predict(vectors)

pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vectors.toarray())

colors = ["r","b","c","y","m"]

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

fig,ax = plt.subplots(figsize=(50,50))

ax.scatter(x_axis,y_axis,c=[colors[d] for d in kmean_indices])

#for i,txt in enumerate(sentiment):
    #ax.annotate(txt,(x_axis[i],y_axis[i]))

plt.savefig("trc.png")