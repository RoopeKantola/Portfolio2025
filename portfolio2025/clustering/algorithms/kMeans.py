import math
import random

import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from jinja2 import Template
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import alphashape
import shapely

random.seed(1)


def kMeans(dataframe, no_of_clusters):
    go_frames = []

    data = dataframe.iloc[:, 0:4]
    labels = dataframe.iloc[:, -1]

    random_cluster_centers = []
    for i in range(no_of_clusters):
        random_cluster_centers.append([random.uniform(-10, 10), random.uniform(-10, 10)])

    print(random_cluster_centers)

    random_cluster_std = []
    random_std = random.uniform(0.8, 2.5)
    for i in range(no_of_clusters):
        random_cluster_std.append(random_std)



    X, y = make_blobs(n_samples=300, cluster_std=random_cluster_std,
                      centers=random_cluster_centers, n_features=2, random_state=1)

    data = pd.DataFrame(X, columns=["x", "y"])
    y = y.tolist()
    for i, center in enumerate(y):
        y[i] = "center" + str(center)

    #print(y)

    labels = pd.Series(y)

    ss = StandardScaler()
    normalized_data = ss.fit_transform(data)

    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(normalized_data)
    transformed_data = pd.DataFrame(transformed_data, columns=["PC1", "PC2"])

    transformed_data["labels"] = labels

    centers = calculate_centers(transformed_data, first_assignment=True, no_of_clusters=no_of_clusters)

    distance_matrix = calculate_distances(transformed_data, centers)
    transformed_data = assign_centers(transformed_data, distance_matrix)

    previous_centers = pd.DataFrame(columns=["x", "y"], index=[i for i in range(no_of_clusters)])

    # Adding a list of symbols and colors for use
    symbols = ["diamond", "circle", "star", "triangle-up", "hexagon", "octagon", "square", "hexagram",
               "star-square", "star-diamond", "hourglass", "bowtie", "circle-x", "square-x", "x-thin"]

    colors = ["red", "blue", "orange", "magenta", "cyan", "maroon", "palegreen",
              "coral", "deeppink", "crimson", "lightblue",
              "orchid", "violet", "yellow", "skyblue", "mintcream"]

    keys = pd.Series.unique(transformed_data["labels"])
    label_to_symbol_dict = {keys[i]: symbols[i] for i in range(len(keys))}
    keys = pd.Series.unique(transformed_data["center"])
    center_to_color_dict = {keys[i]: colors[i] for i in range(len(keys))}

    fig = go.Figure()

    plots = [go.Scatter(x=transformed_data['PC1'],
                        y=transformed_data['PC2'],
                        mode='markers',
                        marker_color=transformed_data["center"].map(center_to_color_dict),
                        marker_symbol=transformed_data["labels"].map(label_to_symbol_dict),

                        ),
             ]

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x")

    # Add centers to plot
    center_plot = go.Scatter(
        x=centers["x"],
        y=centers["y"],
        name="centers",
        mode="markers",
        marker=dict(size=15, color="black", symbol="cross", line=dict(color="gray", width=2)))
    fig.add_trace(center_plot)

    for plot in plots:
        fig.add_trace(plot)

    go_frames.append(go.Frame(data=[center_plot] + plots))
    max_iter = 20
    iter = 0

    while not np.array_equal(centers.to_numpy(), previous_centers.to_numpy()) and iter < max_iter:
        iter += 1
        print("iter:", iter)
        previous_centers = pd.DataFrame.copy(centers)
        centers = calculate_centers(data=transformed_data, first_assignment=False,
                                    no_of_clusters=no_of_clusters, centers=centers.to_numpy())
        #print("CEENTERRS", centers)
        distance_matrix = calculate_distances(transformed_data, centers)
        transformed_data = assign_centers(transformed_data, distance_matrix)

        plots = [go.Scatter(x=transformed_data['PC1'],
                            y=transformed_data['PC2'],
                            mode='markers',
                            marker_color=transformed_data["center"].map(center_to_color_dict),
                            marker_symbol=transformed_data["labels"].map(label_to_symbol_dict),

                            )]

        go_frames.append(go.Frame(data=[go.Scatter(
            x=centers["x"],
            y=centers["y"],
            name="centers",
            mode="markers",
            marker=dict(size=15, color="black", symbol="cross", line=dict(color="gray", width=2)))] + plots))

    fig.update(frames=go_frames)

    updatemenus = [dict(
        buttons=[
            dict(
                args=[None, {"frame": {"duration": 500, "redraw": True},
                             "fromcurrent": True, "transition": {"duration": 800}}],
                label="Play",

                method="animate"
            ),

        ],
        direction="left",
        pad={"r": 10, "t": 87},
        showactive=False,
        type="buttons",
        x=0.1,
        xanchor="right",
        y=0,
        yanchor="top"
    )]

    fig.update_layout(
        title_text="Kmeans clustering",
        title_x=0.5,
        width=840,
        height=680,
        updatemenus=updatemenus,
    )

    fig.show()

    datafile = open("datafile.txt", "w")
    datafile.write(transformed_data.to_csv())
    datafile.close()

    return True


def calculate_centers(data, first_assignment=False, no_of_clusters=2, centers=np.array([])):
    min_x = min(data["PC1"])
    max_x = max(data["PC1"])
    min_y = min(data["PC2"])
    max_y = max(data["PC2"])

    new_centers = []
    if first_assignment:
        for i in range(no_of_clusters):
            #print("i:", i)
            center = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            new_centers.append(center)
    else:
        for i, _ in enumerate(centers):

            center = f"center{i}"

            means = data[data["center"] == center].mean(numeric_only=True)

            if not np.isnan(means.to_numpy()).any():
                new_centers.append(means.to_numpy())
            else:
                new_centers.append(centers[i])

    new_centers = pd.DataFrame(new_centers, columns=["x", "y"])
    #print("centers:", new_centers)

    return new_centers


def euclidean_distance(point1, point2):
    sum = 0
    for dimension, _ in enumerate(point1):
        sum += (point2[dimension] - point1[dimension]) ** 2

    distance = math.sqrt(sum)
    return distance


def calculate_distances(dataframe, centers):
    distances = pd.DataFrame()

    for id, _ in enumerate(centers.iterrows()):
        distances[f"center{id}"] = ''
        for row, _ in enumerate(dataframe.iterrows()):
            distances.loc[row, f"center{id}"] = euclidean_distance(
                dataframe.to_numpy()[row, :2], centers.to_numpy()[id])


    #print("distances:", distances)

    return distances


def assign_centers(dataframe, distance_matrix):
    dataframe["center"] = distance_matrix.idxmin(axis=1)

    return dataframe


if __name__ == "__main__":
    dataframe = pd.read_csv("../../datasets/iris/iris.data")

    kMeans(dataframe, 3)
