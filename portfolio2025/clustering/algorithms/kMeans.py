import math
import random

import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go
from jinja2 import Template
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import alphashape
import shapely

random.seed(1)


def kMeans(dataframe, no_of_clusters):
    frames = []
    frame = []

    data = dataframe.iloc[:, 0:4]
    labels = dataframe.iloc[:, -1]

    ss = StandardScaler()
    normalized_data = ss.fit_transform(data)

    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(normalized_data)
    transformed_data = pd.DataFrame(transformed_data, columns=["PC1", "PC2"])

    transformed_data["labels"] = labels

    centers = calculate_centers(transformed_data, first_assignment=True, no_of_clusters=3)

    distance_matrix = calculate_distances(transformed_data, centers)
    transformed_data = assign_centers(transformed_data, distance_matrix)

    frame.append(transformed_data)
    frame.append(centers)

    print(frame)

    fig = go.Figure()
    fig_2 = px.scatter(transformed_data, x="PC1", y="PC2", color="center", symbol="labels",
                       symbol_sequence=["star", "circle", "diamond"])

    for plot in fig_2.data:
        fig.add_trace(plot)

    fig.update_traces(marker=dict(size=10))
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x")

    fig.add_trace(go.Scatter(
        x=centers["x"],
        y=centers["y"],
        name="centers",
        mode="markers",
        marker=dict(size=15, color="black", symbol="cross", line=dict(color="cyan", width=2))))

    fig.show()

    calculate_centers(data=transformed_data, first_assignment=False, no_of_clusters=3)

    return True


def calculate_centers(data, first_assignment=False, no_of_clusters=2):
    min_x = min(data["PC1"])
    max_x = max(data["PC1"])
    min_y = min(data["PC2"])
    max_y = max(data["PC2"])

    centers = []
    if first_assignment:
        for i in range(no_of_clusters):
            center = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            centers.append(center)
    else:
        for i, center in enumerate(pd.Series.unique(data["center"])):
            print(center)
            means = data[data["center"] == center].mean(numeric_only=True)
            print(means)
            print(means.to_numpy())

    centers = pd.DataFrame(centers, columns=["x", "y"])

    return centers


def euclidean_distance(point1, point2):
    sum = 0
    for dimension, _ in enumerate(point1):
        sum += (point2[dimension] - point1[dimension]) ** 2

    distance = math.sqrt(sum)
    return distance


def calculate_distances(dataframe, centers):
    distances = pd.DataFrame()

    for id, _ in enumerate(centers.iterrows()):
        distances[f"center{id + 1}"] = ''
        for row, _ in enumerate(dataframe.iterrows()):
            distances.loc[row, f"center{id + 1}"] = euclidean_distance(
                dataframe.to_numpy()[row, :2], centers.to_numpy()[id])

    return distances


def assign_centers(dataframe, distance_matrix):
    dataframe["center"] = distance_matrix.idxmin(axis=1)

    return dataframe


def animate_kMeans(dataframe):
    return True


if __name__ == "__main__":
    dataframe = pd.read_csv("../../datasets/iris/iris.data")

    kMeans(dataframe, 3)
