import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame(np.random.randint(0, 100, size=(100, 2)))
print(df.head())





