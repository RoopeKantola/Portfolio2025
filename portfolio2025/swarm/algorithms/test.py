import networkx as nx
import numpy as np

G = nx.Graph()

G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(2, 5)
G.add_edge(1, 4)

population = np.array([0, 1, 2, 3, 4])
fitnesses = np.array([1, 0.9, 0.8, 0.7, 0.6])


# Tournament selection is used
def select_parent(population, fitnesses, k=3):
    candidates = np.random.choice(len(population), k, replace=True)
    print(candidates)
    print(fitnesses[candidates])
    return np.argmax(fitnesses[candidates])

#np.random.seed(1)


'''
print(select_parent(population, fitnesses))

parent_a = [0,1,2, 3, 4, 5, 6, 7, 8, 9]
parent_b = np.random.permutation(parent_a)

print(parent_b)

slice_point_1 = np.random.randint(0, len(parent_a)-1)
slice_point_2 = np.random.randint(slice_point_1+1, len(parent_a))
child = [np.nan]*len(parent_a)
child[slice_point_1:slice_point_2] = parent_a[slice_point_1:slice_point_2]

j = 0
for i in range(len(parent_b)):
    if parent_b[i] not in child:
        while not np.isnan(child[j]):
            print(child[j])
            j += 1
        child[j] = int(parent_b[i])
        j += 1


print(child)
'''

i, j = np.random.randint(0, len([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 2)

a = [0, 1, 2, 3, 4]
b = [4, 2, 1, 3, 0]

sorted_a, sorted_b = zip(*sorted(zip(b, a)))

x = zip(b, a)

current_tour = [0, 1, 2, 3, 4]

delta = 15
temperature = 3

print(np.exp(-delta / temperature))

for i in range(1):
    print(np.random.random())

