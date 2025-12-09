# Simulated annealing for TSP
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import ast

plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\roope\Ohjelmat\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe'

np.random.seed(999)


class SimulatedAnnealing():
    def __init__(self, Graph, initial_temperature, iterations_per_temperature, cooling_rate, max_main_iterations,
                 temp_threshold, print_iterations=False, print_frequency=10):
        self.Graph = Graph
        self.max_main_iterations = max_main_iterations
        self.temp_threshold = temp_threshold
        self.initial_temperature = initial_temperature
        self.temperature = self.initial_temperature
        self.iterations_per_temperature = iterations_per_temperature
        self.cooling_rate = cooling_rate
        self.print_iterations = print_iterations
        self.print_requency = print_frequency
        self.N = len(Graph.nodes)
        self.distance_matrix = np.full((self.N, self.N), fill_value=np.inf)
        for u, v, data in self.Graph.edges(data=True):
            if u != v:
                self.distance_matrix[u - 1][v - 1] = int(data['weight'])
                self.distance_matrix[v - 1][u - 1] = int(data['weight'])

        self.current_tour = self.random_tour()
        self.current_length = self.tour_length(self.current_tour)
        self.best_tour = self.current_tour
        self.best_length = self.current_length
        self.best_length_history = []
        self.average_length_history = []
        self.best_tour_history = []

    def tour_length(self, tour):
        length = 0
        for i in range(len(tour) - 1):
            length += self.distance_matrix[int(tour[i]), int(tour[i + 1])]
        length += self.distance_matrix[int(tour[-1]), int(tour[0])]
        return length

    def random_tour(self):
        tour = np.array(np.random.permutation(list(self.Graph)))
        tour -= 1
        return tour

    def generate_neighbor(self):
        cut_point_1 = np.random.randint(0, len(self.current_tour))
        cut_point_2 = np.random.randint(cut_point_1 + 1, len(self.current_tour) + 1)
        neighbor = np.concatenate((self.current_tour[:cut_point_1],
                                   self.current_tour[cut_point_1:cut_point_2][::-1],
                                   self.current_tour[cut_point_2:]), axis=0)
        return np.array(neighbor, dtype=int), cut_point_1, cut_point_2

    def neighbor_delta(self, cut_point_1, cut_point_2):
        n = len(self.current_tour)
        a = int(self.current_tour[cut_point_1 - 1])
        b = int(self.current_tour[cut_point_1])
        c = int(self.current_tour[cut_point_2 - 1])
        d = int(self.current_tour[cut_point_2 % n])

        removed = self.distance_matrix[a, b] + self.distance_matrix[c, d]
        added = self.distance_matrix[a, c] + self.distance_matrix[b, d]

        return added - removed

    def run_sa(self):
        main_iterations = 0
        while True:
            accepted_moves = 0
            self.best_length_history.append(self.best_length)
            self.best_tour_history.append(self.best_tour)
            for _ in range(self.iterations_per_temperature):
                new_tour, cut_point_1, cut_point_2 = self.generate_neighbor()
                delta = self.neighbor_delta(cut_point_1, cut_point_2)

                if delta < 0 or np.random.random() < np.exp(-delta / self.temperature):
                    accepted_moves += 1
                    #print("old length", self.current_length, "old tour length", self.tour_length(self.current_tour), self.current_tour)
                    self.current_length += delta
                    self.current_tour = new_tour
                    #print("new length", self.current_length, "new tour length", self.tour_length(self.current_tour), self.current_tour)

                if self.current_length < self.best_length:
                    self.best_length = self.current_length
                    self.best_tour = self.current_tour

                if accepted_moves % 100 == 0:  # every 100 accepts (or every 1000 proposals)
                    full = self.tour_length(self.current_tour)
                    if abs(full - self.current_length) > 1e-9:
                        print("Length mismatch! full:", full, "current:", self.current_length)
                        raise RuntimeError("Length drift detected")

            self.temperature *= self.cooling_rate
            main_iterations += 1

            if main_iterations % self.print_requency == 0:
                print(f"Iteration {main_iterations}: best length: {self.best_length}")
            if self.temperature < self.temp_threshold:
                break
            if main_iterations > self.max_main_iterations:
                break
        return self.best_length, self.best_tour, self.best_length_history, self.best_tour_history

def plot_convergence(Graph, best_length_history):
    print("Plotting convergence...")
    fig, ax = plt.subplots()
    ax.set_title(f"\nSimulated annealing convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tour length")
    ax.set_xlim(0, len(best_length_history))
    ax.set_ylim(0, max(best_length_history) * 1.1)

    (best_line,) = ax.plot([], [], label=f"Best tour length: {best_length_history[0]:.0f}", color="blue")
    ax.legend()

    def update(frame):
        best_line.set_data(range(frame), best_length_history[:frame])
        best_line.set_label(f"Best tour length: {best_length_history[frame]:.0f}")
        ax.legend([best_line],
                  [best_line.get_label()])
        return best_line,

    ani = FuncAnimation(fig, update, frames=len(best_length_history), interval=5, blit=True)
    plt.tight_layout()
    ani.save(f"../static/videos/SA_convergence_{Graph.name}.mp4", writer="ffmpeg", fps=30, dpi=200)
    print("Animation saved.")

def plot_best_tour(Graph, best_tour_history):
    print("Plotting best tour...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"\nBest tour development")
    pos = nx.get_node_attributes(Graph, 'pos')
    nodes = nx.draw_networkx_nodes(Graph, pos, node_size=200, ax=ax)
    labels = nx.draw_networkx_labels(Graph, pos, font_size=8, ax=ax)
    edges = list(Graph.edges())

    def update(frame):
        ax.clear()
        nx.draw_networkx_nodes(Graph, pos, node_size=200, ax=ax)
        nx.draw_networkx_labels(Graph, pos, font_size=8, ax=ax)
        tour = best_tour_history[frame]
        tour_edges = []
        bg_edges = []
        for edge in range(len(tour)-1):
            tour_edges.append((int(tour[edge]+1), int(tour[edge+1])+1))
        tour_edges.append((int(tour[-1]+1), int(tour[0]+1)))

        tour_edge_set = frozenset(tour_edges)
        for edge in Graph.edges():
            if edge not in tour_edge_set:
                bg_edges.append(edge)

        nx.draw_networkx_edges(
            G, pos,
            edgelist=bg_edges,
            edge_color="gray",
            width=2,
            alpha=0.5,
            ax=ax
        )
        nx.draw_networkx_edges(
            G, pos,
            edgelist=tour_edges,
            edge_color="yellow",
            width=4,
            alpha=0.8,
            ax=ax
        )

        ax.set_title(f"Best tour - iteration {frame}")
        ax.axis('off')

    ani = FuncAnimation(fig, update, frames=len(best_tour_history), interval=5, blit=False)
    plt.tight_layout()
    ani.save(f"../static/videos/SA_best_tour_{Graph.name}.mp4", writer="ffmpeg", fps=30, dpi=200)
    print("Animation saved.")

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
        if len(G.nodes) < 25:
            max_main_iterations = 500
        else:
            max_main_iterations = 1000
        curr_best = np.inf
        curr_tours = []
        curr_lengths = []

        for i in range(30):
            print("\n:", i)
            sa = SimulatedAnnealing(Graph=G, initial_temperature=1000, cooling_rate=0.995, iterations_per_temperature=200,
                                    max_main_iterations=max_main_iterations, temp_threshold=1e-3, print_iterations=True,
                                    print_frequency=100)
            best_length, best_tour, best_length_history, best_tour_history = sa.run_sa()
            if best_length < curr_best:
                print(f"New best, i={i}, best length: {best_length}")
                curr_best = best_length
                curr_tours = best_tour_history
                curr_lengths = best_length_history
            if best_length == solution:
                print("Breaking")
                break

        #Plot the animations
        plot_convergence(Graph=G, best_length_history=curr_lengths)
        plot_best_tour(Graph=G, best_tour_history=curr_tours)

        print(f"Final results: {int(curr_best)}")
        print(f"Best tour: {best_tour}")
        print(f"Optimal length: {solution}\nDifference: {((curr_best / solution) - 1) * 100:.2F}%")
