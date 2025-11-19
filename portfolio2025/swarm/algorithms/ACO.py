# Ant colony optimization algorithm

import networkx as nx
import numpy.random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import ast

plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\roope\Ohjelmat\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe'


np.random.seed(999)

class AntColony():

    def __init__(self, Graph, number_of_ants=100, iterations=100,
                 alpha=0.5, beta=0.5, rho=0.5,
                 print_iterations=False, print_frequency=10):
        self.print_frequency = print_frequency
        self.print_iterations = print_iterations
        self.Graph = Graph
        self.number_of_ants = number_of_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.best_tour = []
        self.best_length = np.inf
        self.best_length_history = []
        self.average_length_history = []
        self.N = len(Graph.nodes)
        self.distance_matrix = np.full((self.N, self.N), fill_value=np.inf)
        for u, v, data in self.Graph.edges(data=True):
            if u != v:
                self.distance_matrix[u - 1][v - 1] = int(data['weight'])
                self.distance_matrix[v - 1][u - 1] = int(data['weight'])

        self.heuristic_matrix = 1 / self.distance_matrix
        self.pheromone_matrix = np.ones((self.N, self.N), dtype=np.float32)
        self.pheromone_history = [self.pheromone_matrix.copy()]
        valid_mask = ~np.isinf(self.distance_matrix) & ~np.eye(self.distance_matrix.shape[0], dtype=bool)
        self.pheromone_deposit_factor = np.mean(self.distance_matrix[valid_mask])



    def choose_next_location(self, curr_location, visited_mask):
        allowed_mask = ~visited_mask.copy()
        allowed_mask[curr_location] = False

        allowed_indices = np.nonzero(allowed_mask)[0]

        tau_values = self.pheromone_matrix[curr_location, allowed_mask]
        heuristic_values = self.heuristic_matrix[curr_location, allowed_mask]

        desirability = np.power(tau_values, self.alpha) * np.power(heuristic_values, self.beta)


        if desirability.sum() <= 0:
            choice = np.random.choice(allowed_indices)
        else:
            probs = desirability / desirability.sum()
            choice = np.random.choice(allowed_indices, p=probs)

        return choice

    def update_pheromones(self, tours, lengths):
        pheromone_deposits = np.zeros((self.N, self.N))
        for i, tour in enumerate(tours):
            deposit_amount = self.pheromone_deposit_factor/lengths[i]
            for j in range(len(tour)-1):
                a = tour[j]
                b = tour[j + 1]
                pheromone_deposits[a, b] += deposit_amount
                pheromone_deposits[b, a] += deposit_amount

        self.pheromone_matrix *= (1-self.rho)
        self.pheromone_matrix += pheromone_deposits

        #For plotting purposes
        self.pheromone_history.append(self.pheromone_matrix.copy())


    def construct_tour(self):
        start_city = np.random.randint(self.N)
        tour = [start_city]
        total_length = 0
        visited_mask = np.zeros(self.N, dtype=bool)
        visited_mask[start_city] = True

        current_city = start_city
        while not np.all(visited_mask):
            choice = self.choose_next_location(current_city, visited_mask)
            tour.append(choice)
            total_length += self.distance_matrix[current_city][choice]
            visited_mask[choice] = True
            current_city = choice

        tour.append(start_city)
        total_length += self.distance_matrix[current_city][start_city]

        return tour, total_length

    def construct_all_tours(self):
        for i in range(1, self.iterations+1):
            tours = []
            lengths = []

            for ant in range(self.number_of_ants):
                tour, length = self.construct_tour()
                tours.append(tour)
                lengths.append(length)
                if length < self.best_length:
                    self.best_length = length
                    self.best_tour = tour.copy()
            self.update_pheromones(tours, lengths)

            # Visualization step
            self.average_length_history.append(np.mean(lengths))
            self.best_length_history.append(self.best_length)

            if i % self.print_frequency == 0 and self.print_iterations:
                print(f'Iteration number {i}, best length: {self.best_length}.')

        return self.best_tour, self.best_length, self.average_length_history

    def plot_best_lengths(self):

        fig, ax = plt.subplots()
        ax.set_title(f"\nACO convergence\nalpha: {self.alpha}, beta: {self.beta}, rho: {self.rho}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Tour length")
        ax.set_xlim(0, len(self.best_length_history))
        ax.set_ylim(0, max(self.average_length_history)*1.1)

        (best_line, ) = ax.plot([], [], label="Best tour length", color="blue")
        (avg_line, ) = ax.plot([], [], label="Average tour length", color="orange", alpha=0.7)
        ax.legend()

        def update(frame):
            best_line.set_data(range(frame), self.best_length_history[:frame])
            avg_line.set_data(range(frame), self.average_length_history[:frame])
            return best_line, avg_line

        ani = FuncAnimation(fig, update, frames=len(self.best_length_history), interval=100, blit=True)
        plt.tight_layout()
        ani.save(f"../static/videos/ACO_lines_{self.Graph.name}.mp4", writer="ffmpeg", fps=30, dpi=200)

    def plot_pheromone_graph(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"\nPheromone intensity\nalpha: {self.alpha}, beta: {self.beta}, rho: {self.rho}")
        pos = nx.get_node_attributes(G, 'pos')
        nodes = nx.draw_networkx_nodes(self.Graph, pos, node_size=200, ax=ax)
        labels = nx.draw_networkx_labels(self.Graph, pos, font_size=8, ax=ax)
        max_pher_all_times = np.max([np.max(pher_matrix) for pher_matrix in self.pheromone_history])
        edges = list(G.edges())

        def update(frame):

            ax.clear()
            pher = self.pheromone_history[frame]
            max_pher_this_frame = np.max(pher)
            values = [(pher[i - 1, j - 1] + pher[j - 1, i - 1]) / (2 * max_pher_all_times) for i, j in edges]
            current_values = [(pher[i - 1, j - 1] + pher[j - 1, i - 1]) / (2 * max_pher_this_frame) for i, j in edges]
            best_edges = [(self.best_tour[i], self.best_tour[i+1]) for i in range(len(self.best_tour)-1)]
            nx.draw_networkx_nodes(G, pos, node_size=200, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                edge_color=values,
                edge_cmap=plt.colormaps["plasma"],
                width=[2 + 5 * v for v in values],
                alpha=current_values,
                ax=ax
            )
            #nx.draw_networkx_edges(G, pos, edgelist=best_edges, edge_color="green", alpha=1, ax=ax, width=1)
            ax.set_title(f"Pheromone levels - iteration {frame}")
            ax.axis('off')

        ani = FuncAnimation(fig, update, frames=len(self.pheromone_history), interval=500, blit=False)
        plt.tight_layout()
        ani.save(f"../static/videos/ACO_pheromones_{self.Graph.name}.mp4", writer="ffmpeg", fps=10, dpi=200)


if __name__ == "__main__":

    '''
    Problem and optimal solution data set up 
    '''
    file_names_not_used_currently = ["eil51"]
    file_names = ["burma14", "att48"]

    '''
    Solution from the solution file
    '''
    for file_name in file_names:

        with open('../../datasets/tsp_graphs/solutions') as f:
            solutions_text = f.read()

        solutions_text = solutions_text.replace(" ", "")
        solutions_list = solutions_text.splitlines()
        n = 1
        solutions = dict()
        for entry in solutions_list:
            key, _, value = entry.partition(":")
            solutions[key] = int(value)

        solution = solutions[file_name]

        '''
        Getting the graph data from the graph file
        '''
        G = nx.Graph()
        with open(f'../../datasets/tsp_graphs/{file_name}.txt', 'r') as f:
            data = ast.literal_eval(f.read())

        edges, coordinates = data
        coordinates_dict = coordinates[0]

        for node, (x, y) in coordinates_dict.items():
            G.add_node(node, pos=(x, y))

        for source, sink, weight in edges:
            G.add_edge(source, sink, weight=weight)

        G.name = file_name
        print(file_name)

        test_suite = {"alpha": [1, 2, 3],
                      "beta": [2, 3, 4],
                      "rho": [0.1, 0.2, 0.3]}
        results = {}
        i = 1
        for alpha in test_suite["alpha"]:
            for beta in test_suite["beta"]:
                for rho in test_suite["rho"]:
                    print(i)
                    i+=1
                    current_setup = (alpha, beta, rho)
                    aco = AntColony(Graph=G, number_of_ants=50,
                                    iterations=150,
                                    alpha=alpha, beta=beta, rho=rho,
                                    print_iterations=False, print_frequency=50)
                    best_tour, best_length, average_length_history = aco.construct_all_tours()
                    avg_length = np.mean(average_length_history[-1])

                    results[current_setup] = {"total": best_length + avg_length, "best": best_length, "avg": avg_length}

        totals = [item["total"] for item in results.values()]
        bests = [item["best"] for item in results.values()]
        avgs = [item["avg"] for item in results.values()]

        for setup in results.keys():
            if results[setup]["best"] == np.min(bests):
                print(f"best setup change, best best")
                best_setup = setup
            elif results[setup]["best"] == np.min(bests) and results[setup]["avg"] < results[best_setup]["avg"]:
                print(f"best setup change, best avg")
                best_setup = setup
        print(f"{best_setup} total: {results[best_setup]}")

        print(f"{file_name} results: {results}")
        with open(f"../static/text_files/results_{file_name}.txt", "w")as f:
            f.writelines(str(best_setup))




        def plot_results():

            fig, ax = plt.subplots()
            ax.set_title("Test suite results")
            ax.set_xlabel("Test setup")
            ax.set_ylabel("Lengths")
            ax.set_xlim(0, len(totals))
            ax.set_ylim(0, max(totals) * 1.1)

            (best_line,) = ax.plot([], [], label="Best tour length", color="blue")
            (avg_line,) = ax.plot([], [], label="Average tour length", color="orange", alpha=0.7)
            (total_line,) = ax.plot([], [], label="Total tour length", color="olive", alpha=0.7)
            ax.legend()

            def update(frame):
                best_line.set_data(range(frame), bests[:frame])
                avg_line.set_data(range(frame), avgs[:frame])
                total_line.set_data(range(frame), totals[:frame])
                return best_line, avg_line, total_line

            ani = FuncAnimation(fig, update, frames=len(totals), interval=100, blit=True)
            plt.tight_layout()
            ani.save(f"../static/videos/results_lines_{file_name}.mp4", writer="ffmpeg", fps=10, dpi=200)

        plot_results()

        print(file_name, best_setup)
        best_best = np.inf
        best_avg = np.inf
        for i in range(10):
            aco = AntColony(Graph=G, number_of_ants=100,
                            iterations=300,
                            alpha=best_setup[0], beta=best_setup[1], rho=best_setup[2],
                            print_iterations=True, print_frequency=20)
            best_tour, best_length, average_length_history = aco.construct_all_tours()
            avg_length = np.mean(average_length_history[-1])
            print(f"{i}: best_length: {best_length}")
            if best_length < best_best:
                best_best = best_length
                best_avg = avg_length
                print(f"Best best found! i={i}")
                aco.plot_best_lengths()
                aco.plot_pheromone_graph()
                print(f"\nBest tour length found: {best_length}")
                print(f"The average tour length is: {avg_length}")
                print(f"The optimal tour length is: {solution}")
                print(f"Difference to best: {((best_length/solution)-1)*100:.2f}% and to average: {((avg_length/solution)-1)*100:.2f}%\n")

            elif best_length == best_best and avg_length < best_avg:
                best_avg = avg_length
                print(f"Best average found! i={i}")
                aco.plot_best_lengths()
                aco.plot_pheromone_graph()
                print(f"\nBest tour length found: {best_length}")
                print(f"The average tour length is: {avg_length}")
                print(f"The optimal tour length is: {solution}")
                print(
                    f"Difference to best: {((best_length / solution) - 1) * 100:.2f}% and to average: {((avg_length / solution) - 1) * 100:.2f}%\n")
