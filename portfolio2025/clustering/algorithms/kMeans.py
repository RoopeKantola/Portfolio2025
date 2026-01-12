import math
import random


random.seed(12)


class kMeans:
    def __init__(self, axis_limit, no_of_clusters, no_of_points):
        self.axis_limit = axis_limit
        self.no_of_clusters = no_of_clusters
        self.no_of_points = no_of_points
        self.iter = 0

    def kMeans_compute(self):
        import pandas as pd
        import numpy as np

        from sklearn.datasets import make_blobs
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        random_cluster_centers = []
        for i in range(self.no_of_clusters):
            random_cluster_centers.append(
                [random.uniform(-self.axis_limit, self.axis_limit), random.uniform(-self.axis_limit, self.axis_limit)])

        random_cluster_std = []
        random_std = random.uniform(0.5, 5)
        for i in range(self.no_of_clusters):
            random_cluster_std.append(random_std)

        X, y = make_blobs(n_samples=self.no_of_points, cluster_std=random_cluster_std,
                          centers=random_cluster_centers, n_features=2, random_state=1)

        data = pd.DataFrame(X, columns=["x", "y"])

        y = y.tolist()
        for i, center in enumerate(y):
            y[i] = "center" + str(center)

        labels = pd.Series(y)

        ss = StandardScaler()
        normalized_data = ss.fit_transform(data)

        pca = PCA(n_components=2)
        transformed_data = pca.fit_transform(normalized_data)
        transformed_data = pd.DataFrame(transformed_data, columns=["PC1", "PC2"])

        transformed_data["labels"] = labels

        centers = self.calculate_centers(transformed_data, first_assignment=True, no_of_clusters=self.no_of_clusters)

        distance_matrix = self.calculate_distances(transformed_data, centers)
        transformed_data = self.assign_centers(transformed_data, distance_matrix)

        previous_centers = pd.DataFrame(columns=["x", "y"], index=[i for i in range(self.no_of_clusters)])

        max_iter = 30

        points_file = open("../static/data/points_file.txt", "w")
        points_file.write(transformed_data.to_csv())
        points_file.close()
        points_file = open("../static/data/points_file.txt", "a")

        centers_file = open("../static/data/centers_file.txt", "w")
        centers_file.write(centers.to_csv())
        centers_file.close()
        centers_file = open("../static/data/centers_file.txt", "a")

        while not np.array_equal(centers.to_numpy(), previous_centers.to_numpy()) and self.iter < max_iter:
            self.iter += 1
            print("iter:", self.iter)
            previous_centers = pd.DataFrame.copy(centers)
            centers = self.calculate_centers(data=transformed_data, first_assignment=False,
                                             no_of_clusters=self.no_of_clusters, centers=centers.to_numpy())

            distance_matrix = self.calculate_distances(transformed_data, centers)
            transformed_data = self.assign_centers(transformed_data, distance_matrix)

            points_file.write(transformed_data.to_csv(header=False))
            centers_file.write(centers.to_csv(header=False))

        centers_file.close()

    def animate_kMeans(self):
        import pandas as pd
        import numpy as np

        from matplotlib import pyplot as plt
        from matplotlib import animation as ani
        plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\roope\Ohjelmat\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe'
        plt.rcParams['axes.titlesize'] = 'medium'


        point_index = 0
        center_index = 0

        points = pd.read_csv("../static/data/points_file.txt", index_col=0)
        centers = pd.read_csv("../static/data/centers_file.txt", index_col=0)

        limit = max([max([max(points["PC1"]),abs(min(points["PC1"]))]),
                       max([max(points["PC2"]), abs(min(points["PC2"]))]),
                       max([max(centers["x"]), abs(min(centers["x"]))]),
                       max([max(centers["y"]), abs(min(centers["y"]))])])

        no_of_frames = int(len(centers)/self.no_of_clusters)

        colors = {'center0': 'tab:blue', 'center1': 'tab:orange', 'center2': 'tab:green', 'center3': 'tab:red',
                  'center4': 'tab:purple', 'center5': 'tab:brown', 'center6': 'tab:pink', 'center7': 'tab:gray',
                  'center8': 'tab:olive', 'center9': 'tab:cyan'}
        figure, axis = plt.subplots()

        axis.set(xlim=[-limit, limit], ylim=[-limit, limit])
        axis.scatter(points["PC1"][point_index:(point_index + self.no_of_points)],
                     points["PC2"][point_index:point_index + self.no_of_points],
                     c=points["center"][point_index:(point_index + self.no_of_points)].map(colors))
        axis.scatter(centers["x"][center_index:(center_index + self.no_of_clusters)],
                     centers["y"][center_index:(center_index + self.no_of_clusters)], marker="*", c="black")


        def update_kMmeans_animation(frame):
            point_index = self.no_of_points * frame
            center_index = self.no_of_clusters * frame
            plt.cla()
            axis.set(xlim=[-limit, limit], ylim=[-limit, limit])
            axis.scatter(points["PC1"][point_index:(point_index + self.no_of_points)],
                         points["PC2"][point_index:point_index + self.no_of_points],
                         c=points["center"][point_index:(point_index + self.no_of_points)].map(colors))
            axis.scatter(centers["x"][center_index:(center_index + self.no_of_clusters)],
                         centers["y"][center_index:(center_index + self.no_of_clusters)], c="black", marker="*")

        anim = ani.FuncAnimation(fig=figure, func=update_kMmeans_animation, frames=no_of_frames, interval=500,
                                 repeat=False)
        anim.save(filename="../static/videos/kmeans.mp4", fps=2)

        plt.close()

    def calculate_centers(self, data, first_assignment=False, no_of_clusters=2, centers=[]):
        import pandas as pd
        import numpy as np

        min_x = min(data["PC1"])
        max_x = max(data["PC1"])
        min_y = min(data["PC2"])
        max_y = max(data["PC2"])

        new_centers = []
        if first_assignment:
            for i in range(no_of_clusters):
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

        return new_centers

    def euclidean_distance(self, point1, point2):
        sum = 0
        for dimension, _ in enumerate(point1):
            sum += (point2[dimension] - point1[dimension]) ** 2

        distance = math.sqrt(sum)
        return distance

    def calculate_distances(self, dataframe, centers):
        import pandas as pd
        import numpy as np

        distances = pd.DataFrame()

        for id, _ in enumerate(centers.iterrows()):
            distances[f"center{id}"] = ''
            for row, _ in enumerate(dataframe.iterrows()):
                distances.loc[row, f"center{id}"] = self.euclidean_distance(
                    dataframe.to_numpy()[row, :2], centers.to_numpy()[id])

        return distances

    def assign_centers(self, dataframe, distance_matrix):
        dataframe["center"] = distance_matrix.idxmin(axis=1)

        return dataframe


if __name__ == "__main__":
    kMeans = kMeans(axis_limit=10, no_of_clusters=10, no_of_points=500)
    kMeans.kMeans_compute()
    kMeans.animate_kMeans()
