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

    output_html_path=r"/graph.html"
    input_template_path = r"/clustering/index.html"

    plotly_jinja_data = {"fig":fig.to_html(full_html=False)}
    #consider also defining the include_plotlyjs parameter to point to an external Plotly.js as described above

    with open(output_html_path, "w", encoding="utf-8") as output_file:
        with open(input_template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render(plotly_jinja_data))


if __name__ == "__main__":
    pass
