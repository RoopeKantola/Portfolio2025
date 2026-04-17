import time

import numpy as np

np.random.seed(1)


def random_swap_kMeans(txt_data, k=2, iterations_per_kMeans=2, max_swaps=500):

    start = time.time_ns()
    # store dataset
    dataset = np.loadtxt(fname=txt_data)

    centroid_idx = np.sort(np.random.choice(dataset.shape[0], k, replace=False))
    centroids = dataset[centroid_idx]
    partition = optimal_partition(dataset, centroids)
    previous_centroids = np.zeros(shape=centroids.shape)

    minimum_sse = compute_sse(dataset, centroids, partition)
    swap_no = 0

    # Initializing history storage for animation frames
    history = [{
        "old_centroids": centroids.copy(),
        "old_partition": partition.copy(),
        "steps": [],
        "accepted": True
    }]

    # main swap loop
    for swap in range(max_swaps):
        swap_no += 1

        # make swap
        new_centroids = random_swap(centroids, dataset)
        # assign new parition
        new_partition = optimal_partition(dataset, new_centroids)
        # Initialization of steps for local refinement animation
        steps = [{
            "centroids": new_centroids.copy(),
            "partition": new_partition.copy()
        }]

        for i in range(iterations_per_kMeans):
            # calculate new centroids
            new_centroids = compute_centroids(dataset, new_partition, k)
            # assign new parition
            new_partition = optimal_partition(dataset, new_centroids)

            steps.append({
                "centroids": new_centroids.copy(),
                "partition": new_partition.copy()
            })

        new_partition = optimal_partition(dataset, new_centroids)

        new_sse = compute_sse(dataset, new_centroids, new_partition)

        # Store history for animation
        history.append({
            "old_centroids": centroids.copy(),
            "old_partition": partition.copy(),
            "steps": steps,
            "accepted": new_sse < minimum_sse
        })

        if new_sse < minimum_sse:
            minimum_sse = new_sse
            centroids = new_centroids.copy()
            partition = new_partition.copy()
            print(f"Improvement on swap no. {swap_no}, sse: {minimum_sse}")

    sse_value = compute_sse(dataset, centroids, partition)

    stop = time.time_ns()
    processing_time_milliseconds = (stop - start) / 1000000
    return centroids, dataset, partition, sse_value, history, processing_time_milliseconds

def build_animation_frames(dataset, history):
    frames = []

    for entry in history:

        # old state
        frames.append({
            "centroids": entry["old_centroids"],
            "partition": entry["old_partition"],
            "status": "current"
        })

        # steps
        for i, step in enumerate(entry["steps"]):

            if i < len(entry["steps"]) - 1:
                status = "candidate"
            else:
                if entry["accepted"]:
                    status = "current"
                else:
                    status = "rejected"

            frames.append({
                "centroids": step["centroids"],
                "partition": step["partition"],
                "status": status
            })

        # revert if rejected
        if not entry["accepted"]:
            frames.append({
                "centroids": entry["old_centroids"],
                "partition": entry["old_partition"],
                "status": "current"
            })

    return frames

def random_swap(centroids, dataset):
    new_centroids = centroids.copy()
    centroid_choice = np.random.randint(new_centroids.shape[0])
    new_centroid_choice = np.random.randint(dataset.shape[0])
    new_centroids[centroid_choice] = dataset[new_centroid_choice]
    return new_centroids


def optimal_partition(dataset, centroids):
    distances = np.linalg.norm(
        dataset[:, None, :] - centroids[None, :, :],
        axis=2
    )
    partition = np.argmin(distances, axis=1)

    return partition


def compute_centroids(dataset, partition, k):
    dimensions = dataset.shape[1]
    centroids = np.zeros((k, dimensions))

    for i in range(k):
        cluster_data = dataset[partition == i]

        if len(cluster_data) > 0:
            centroids[i] = cluster_data.mean(axis=0)

    return centroids


def compute_sse(dataset, centroids, partition):
    diffs = dataset - centroids[partition]
    return np.sum(diffs ** 2)


def format_data_for_vis(dataset, centroids, partition):
    import pandas as pd
    df = pd.DataFrame(dataset, columns=["PC1", "PC2"])
    df["center"] = ["center" + str(int(i)) for i in partition]
    centers_df = pd.DataFrame(centroids, columns=["x", "y"])
    return df, centers_df


def write_final_state(df, centers_df):
    points_file = open("../static/data/randomSwap/points_file.txt", "w")
    points_file.write(df.to_csv())
    points_file.close()

    centers_file = open("../static/data/randomSwap/centers_file.txt", "w")
    centers_file.write(centers_df.to_csv())
    centers_file.close()


def plot_final_state(df, centers_df, partition, save_to_file=False, file_name="default_final_state", title="clustering", k=50):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("hsv", k)

    colors_list = [cmap(int(i)) for i in partition]

    plt.figure()

    # Plot points
    plt.scatter(
        df["PC1"],
        df["PC2"],
        c=colors_list
    )

    # Plot centroids
    plt.scatter(
        centers_df["x"],
        centers_df["y"],
        c="black",
        marker="*",
        s=200
    )

    plt.title(f"{title} (Final State)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    if save_to_file:
        plt.savefig(f"../../static/images/{file_name}.png", dpi=300, bbox_inches="tight")

    plt.show()


def kMeans(txt_data, k):
    start = time.time_ns()

    min_sse = np.inf
    dataset = np.loadtxt(fname=txt_data)

    for _ in range(100):
        centroid_idx = np.sort(np.random.choice(dataset.shape[0], k, replace=False))
        centroids = dataset[centroid_idx]
        partition = optimal_partition(dataset, centroids)
        previous_centroids = np.zeros(shape=centroids.shape)
        converged = False
        while not converged:
            # assign new parition
            partition = optimal_partition(dataset, centroids)

            # calculate new centroids
            previous_centroids = centroids
            centroids = compute_centroids(dataset, partition, k)

            if np.array_equal(centroids, previous_centroids):
                converged = True

        sse = compute_sse(dataset, centroids, partition)
        if sse < min_sse:
            min_sse = sse
            best_partition = partition
            best_centroids = centroids

        stop = time.time_ns()
        processing_time_milliseconds = (stop-start)/1000000

    return best_centroids, dataset, best_partition,processing_time_milliseconds

def animate_random_swap(dataset, frames, save_path, k):
    import pandas as pd
    from matplotlib import pyplot as plt
    from matplotlib import animation as ani
    plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\roope\Ohjelmat\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe'
    plt.rcParams['axes.titlesize'] = 'medium'

    df = pd.DataFrame(dataset, columns=["PC1", "PC2"])

    cmap = plt.get_cmap("hsv", k)

    figure, axis = plt.subplots()

    def update(frame_idx):
        plt.cla()

        frame = frames[frame_idx]
        centroids = frame["centroids"]
        partition = frame["partition"]
        status = frame["status"]

        x_min, x_max = df["PC1"].min(), df["PC1"].max()
        y_min, y_max = df["PC2"].min(), df["PC2"].max()

        padding = 0.1

        x_range = x_max - x_min
        y_range = y_max - y_min

        axis.set(
            xlim=[x_min - padding * x_range, x_max + padding * x_range],
            ylim=[y_min - padding * y_range, y_max + padding * y_range]
        )

        colors_list = [cmap(int(i)) for i in partition]

        axis.scatter(df["PC1"], df["PC2"], c=colors_list)

        if status == "rejected":
            axis.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x", s=200)
        else:
            axis.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="*", s=200)

    anim = ani.FuncAnimation(
        fig=figure,
        func=update,
        frames=len(frames),
        interval=500,
        repeat=False
    )

    anim.save(filename=save_path, fps=15)
    plt.close()


if __name__ == "__main__":
    file_path = "../../datasets/clustering/a3.txt"
    file_name = "a3"
    #Random swap
    centroids, dataset, partition, sse_value, history, processing_time_milliseconds = random_swap_kMeans(txt_data=file_path, k=50, iterations_per_kMeans=2, max_swaps=430)
    df, centers_df = format_data_for_vis(dataset, centroids, partition)
    write_final_state(df, centers_df)
    plot_final_state(df, centers_df, partition=partition, save_to_file=False, file_name=f"random_swap_"
                                                                                        f"{file_name}", title="Random swap", k=50)

    print(f"milliseconds for random swap: {processing_time_milliseconds}")

    print(len(history))
    print(history[0].keys())
    frames = build_animation_frames(dataset, history)
    animate_random_swap(dataset, frames, f"../../static/videos/random_swap_{file_name}.mp4", k=50)

    #normal k-means
    centroids, dataset, partition, processing_time_milliseconds = kMeans(file_path, 50)
    print(f"milliseconds for k-means: {processing_time_milliseconds}")
    df, centers_df = format_data_for_vis(dataset, centroids, partition)
    plot_final_state(df, centers_df, partition=partition, save_to_file=False, file_name=f"kMeans_{file_name}", title="K-means", k=50)
