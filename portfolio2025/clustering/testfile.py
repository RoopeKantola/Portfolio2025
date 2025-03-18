import pandas as pd
import numpy as np
import plotly.express as px
from jinja2 import Template
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def draw_graph():
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 2)), columns=["a", "b"])

    fig = px.scatter(x=df["a"], y=df["b"])

    fig = fig.to_html()

    return fig


if __name__ == "__main__":
    pass
