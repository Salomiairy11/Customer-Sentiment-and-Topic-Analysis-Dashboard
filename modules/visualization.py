import matplotlib as plt
from sklearn.decomposition import PCA

kmean_indices = model.fit_predict(vectors)

pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vectors.toarray())

colors = ["r","b","c","y","m"]

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

fig,ax = plt.subplots(figsize=(50,50))

ax.scatter(x_axis,y_axis,c=[colors[d] for d in kmean_indices])

for i,txt in enumerate(sentiment):
    ax.annotate(txt[0:5],(x_axis[i],y_axis[i]))

plt.savefig("trc.png")