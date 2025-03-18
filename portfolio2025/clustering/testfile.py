import pandas as pd
import numpy as np
import plotly.express as px
from jinja2 import Template
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def draw_graph():
    row = np.arange(1, 101)
    a = np.linspace(1, 100001, num=100)
    b = np.arange(100, 0, -1)

    data = np.stack((row ,a, b), axis=-1)

    df = pd.DataFrame(data, columns=["row_id", "a", "b"])

    print(df)

    fig = px.scatter(data_frame=df, x=df["a"], y=df["b"], animation_frame="row_id",
                     range_x=[1, 100000], range_y=[0, 100],
                     )

    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5


    #fig.show()

    fig = fig.to_html(auto_play=False)


    return fig



if __name__ == "__main__":
    draw_graph()
