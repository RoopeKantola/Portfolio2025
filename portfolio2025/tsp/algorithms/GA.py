
import networkx as nx
import numpy.random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import ast

plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\roope\Ohjelmat\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe'

np.random.seed(999)

class GeneticAlgorithm():
    def __init__(self, Graph, population_size=100, iterations=100, mutation_rate=0.01, print_iterations=False, print_frequency=10):
        self.Graph = Graph
        self.N = len(Graph.nodes)
        self.distance_matrix = np.full((self.N, self.N), fill_value=np.inf)
        for u, v, data in self.Graph.edges(data=True):
            if u != v:
                self.distance_matrix[u - 1][v - 1] = int(data['weight'])
                self.distance_matrix[v - 1][u - 1] = int(data['weight'])
        self.population_size = population_size
        self.iterations = iterations
        self.base_mutation_rate = mutation_rate
        self.max_mutation_rate = 0.3
        self.no_improvement_counter = 0
        self.mutation_rate = mutation_rate
        self.print_iterations = print_iterations
        self.print_frequency = print_frequency
        self.best_tour = []
        self.best_length = np.inf
        self.fitnesses = np.zeros(self.population_size)
        self.best_fitness = 0
        self.population = self.initial_population()
        self.elite_size = np.max([1, int(0.05 * self.population_size)])
        self.patience = 50
        self.best_length_history = []
        self.average_length_history = []
        self.best_tour_history = []


    def tour_length(self, tour):
        length = 0
        for i in range(len(tour)-1):
            length += self.distance_matrix[int(tour[i]), int(tour[i+1])]
        length += self.distance_matrix[int(tour[-1]), int(tour[0])]
        return length

    def random_tour(self):
        tour = np.array(np.random.permutation(list(self.Graph)))
        tour -= 1
        return tour

    def initial_population(self):
        population = []
        for i in range(self.population_size):
            population.append(self.random_tour())
            self.fitnesses[i] = self.fitness(population[i])
            if self.fitnesses[i] < self.best_fitness:
                self.best_fitness = self.fitnesses[i]


        return population

    def fitness(self, tour):
        return 1 / (1 + self.tour_length(tour))

    #Tournament selection is used
    def select_parent(self, population, fitnesses, k=3):
        candidates = np.random.choice(len(population), k, replace=True)
        best_pos = np.argmax(fitnesses[candidates])
        return candidates[best_pos]

    def ordered_crossover(self, parent_a, parent_b):
        slice_point_1 = np.random.randint(0, len(parent_a) - 1)
        slice_point_2 = np.random.randint(slice_point_1 + 1, len(parent_a))
        child = [None] * len(parent_a)
        child[slice_point_1:slice_point_2] = parent_a[slice_point_1:slice_point_2]

        j = 0
        for i in range(len(parent_b)):
            if parent_b[i] not in child:
                while not child[j] is None:
                    j += 1
                child[j] = int(parent_b[i])
                j += 1
        return np.array(child, dtype=int)

    def mutate(self, tour, mutation_rate):
        if np.random.random() < mutation_rate:
            i, j = np.random.randint(0, len(tour), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour

    def next_generation(self):
        top_fitnesses = np.array(np.argsort(self.fitnesses)[::-1][:self.elite_size])
        new_population = [self.population[fit_index] for fit_index in top_fitnesses]
        new_fitnesses = [self.fitnesses[fit_index] for fit_index in top_fitnesses]
        self.best_length = self.tour_length(self.population[top_fitnesses[0]])
        self.best_tour = self.population[top_fitnesses[0]]
        while len(new_population) < self.population_size:
            parent_a = self.population[self.select_parent(population=self.population, fitnesses=self.fitnesses)]
            parent_b = self.population[self.select_parent(population=self.population, fitnesses=self.fitnesses)]
            child = self.ordered_crossover(parent_a, parent_b)
            child = self.mutate(child, mutation_rate=self.mutation_rate)
            new_population.append(child)
            new_fitnesses.append(self.fitness(child))

        self.population = new_population
        self.fitnesses = np.array(new_fitnesses)


    def run_GA(self):
        best_prev = np.inf
        lengths = [self.tour_length(tour) for tour in self.population]
        self.best_length_history.append(min(lengths))
        self.average_length_history.append(sum(lengths) / len(lengths))
        self.best_tour_history.append(self.population[np.argmin(np.array(lengths))])

        for iteration in range(self.iterations):
            self.next_generation()

            if self.best_length < best_prev:
                best_prev = self.best_length

                self.no_improvement_counter = 0
                self.mutation_rate = max(self.base_mutation_rate,
                                         self.mutation_rate*0.8)
            else:
                self.no_improvement_counter += 1
                if self.no_improvement_counter >= self.patience:
                    self.mutation_rate = min(
                        self.max_mutation_rate,
                        self.mutation_rate * 1.5
                    )
                    self.no_improvement_counter = 0

            lengths = [self.tour_length(tour) for tour in self.population]
            self.best_length_history.append(min(lengths))
            self.average_length_history.append(sum(lengths) / len(lengths))
            self.best_tour_history.append(self.population[np.argmin(np.array(lengths))])

            if self.print_iterations and iteration % self.print_frequency == 0:
                print(f"Iteration {iteration}:\nbest length = {self.best_length}, mutation rate: {self.mutation_rate}")
        return self.best_length, self.best_tour

    def plot_convergence(self):
        print("Plotting convergence...")
        fig, ax = plt.subplots()
        ax.set_title(f"\nGA convergence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Tour length")
        ax.set_xlim(0, len(self.best_length_history))
        ax.set_ylim(0, max(self.average_length_history) * 1.1)

        (best_line,) = ax.plot([], [], label=f"Best tour length: {self.best_length_history[0]:.0f}", color="blue")
        (avg_line,) = ax.plot([], [], label=f"Average tour length: {self.average_length_history[0]:.0f}", color="orange", alpha=0.7)
        ax.legend()

        def update(frame):
            best_line.set_data(range(frame), self.best_length_history[:frame])
            avg_line.set_data(range(frame), self.average_length_history[:frame])
            best_line.set_label(f"Best tour length: {self.best_length_history[frame]:.0f}")
            avg_line.set_label(f"Average tour length: {self.average_length_history[frame]:.0f}")
            ax.legend([best_line, avg_line],
                      [best_line.get_label(), avg_line.get_label()])
            return best_line, avg_line

        ani = FuncAnimation(fig, update, frames=len(self.best_length_history), interval=100, blit=True)
        plt.tight_layout()
        ani.save(f"../static/videos/GA_convergence_{self.Graph.name}.mp4", writer="ffmpeg", fps=30, dpi=200)
        print("Animation saved.")

    def plot_best_tour(self):
        print("Plotting best tour...")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"\nBest tour development")
        pos = nx.get_node_attributes(self.Graph, 'pos')
        nodes = nx.draw_networkx_nodes(self.Graph, pos, node_size=200, ax=ax)
        labels = nx.draw_networkx_labels(self.Graph, pos, font_size=8, ax=ax)
        edges = list(self.Graph.edges())

        def update(frame):
            ax.clear()
            nx.draw_networkx_nodes(self.Graph, pos, node_size=200, ax=ax)
            nx.draw_networkx_labels(self.Graph, pos, font_size=8, ax=ax)
            tour = self.best_tour_history[frame]
            tour_edges = []
            bg_edges = []
            for edge in range(len(tour)-1):
                tour_edges.append((int(tour[edge]+1), int(tour[edge+1])+1))
            tour_edges.append((int(tour[-1]+1), int(tour[0]+1)))

            tour_edge_set = frozenset(tour_edges)
            for edge in self.Graph.edges():
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

        ani = FuncAnimation(fig, update, frames=len(self.best_tour_history), interval=500, blit=False)
        plt.tight_layout()
        ani.save(f"../static/videos/GA_best_tour_{self.Graph.name}.mp4", writer="ffmpeg", fps=10, dpi=200)
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

        population_size = 10*G.number_of_nodes()
        print(population_size)

        curr_best = np.inf
        curr_tour = []

        for i in range(10):
            print(f"{i}...")
            ga = GeneticAlgorithm(Graph=G, population_size=population_size, iterations=500, mutation_rate=0.05, print_iterations=False, print_frequency=100)
            best_length, best_tour = ga.run_GA()
            if best_length < curr_best:
                print(f"New best, i={i}, best length: {best_length}")
                curr_best = best_length
                curr_tour = best_tour
                ga.plot_convergence()
                ga.plot_best_tour()
            if best_length == solution:
                print("Breaking")
                break

        print(f"Final results: {int(curr_best)}")
        print(f"Optimal length: {solution}\nDifference: {((curr_best/solution)-1)*100:.2F}%")
