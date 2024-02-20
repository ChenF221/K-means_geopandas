import matplotlib.pyplot as plot
import pandas as pd
import sklearn.cluster as skl
from yellowbrick.cluster import KElbowVisualizer, silhouette_visualizer

dataframe = pd.read_csv("countries.csv")

# Obtiene las caracteristicas de interés
X = dataframe[["Life Expectancy", "GDP Per Capita", "CO2 Emissions Per Capita"]]

for k in range(2, 11):
    silhouette_visualizer(skl.KMeans(k, random_state=42), X, colors="yellowbrick")

# El Codo


model = skl.KMeans()
visualizer = KElbowVisualizer(model, k=(1, 11))

visualizer.fit(X)
visualizer.show()

plot.show()


k = int(input("Indique el número de clusters k: "))

# Inicialización para algoritmo de clustering con el parámetro K indicado por el usuario
kmeansModel = skl.KMeans(n_clusters=k)

# Carga los datos
kmeansModel.fit(X)

# Obtiene los centroides
centroides = kmeansModel.cluster_centers_


# Obtiene una lista con las labels de los datos
labels = kmeansModel.predict(X)


# Agrega al frame de datos una columna para las labels de clasificación
dataframe["Label"] = labels

# Visualización de 2 clusters
fig = plot.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("Life Expectancy")
ax.set_ylabel("GDP Per Capita")
ax.set_zlabel("CO2 Emissions Per Capita")

ax.scatter(
    dataframe.loc[dataframe["Label"] == 0, "Life Expectancy"],
    dataframe.loc[dataframe["Label"] == 0, "GDP Per Capita"],
    dataframe.loc[dataframe["Label"] == 0, "CO2 Emissions Per Capita"],
    c="springgreen",
)
ax.scatter(
    dataframe.loc[dataframe["Label"] == 1, "Life Expectancy"],
    dataframe.loc[dataframe["Label"] == 1, "GDP Per Capita"],
    dataframe.loc[dataframe["Label"] == 1, "CO2 Emissions Per Capita"],
    c="firebrick",
)

plot.show()

dataframe.to_csv("p10_labeled_countries.csv", encoding="ISO-8859-1")
