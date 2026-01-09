import numpy as np


# Setting of random seed for repeatability
np.random.seed(300)

# Initialize best global cost and position
best_cost_g = np.inf
best_location_g = [np.inf, np.inf]


# Implementation of the Schwefel cost function
def schwefel(position_vector):
    d = len(position_vector)
    total = position_vector[0] * np.sin(np.sqrt(np.abs(position_vector[0]))) + position_vector[1] * np.sin(
        np.sqrt(np.abs(position_vector[1])))
    return 418.9829 * d - total


# Implementation of the Banana cost function
def banana(position_vector):
    return (1 - position_vector[0]) ** 2 + 100 * (position_vector[1] - position_vector[0] ** 2) ** 2


'''
Boids are the individual particles that the PSO uses to find the global minimum.
Boids start with random position within the axis limits and a random velocity between [-1, -1] and [1, 1].
The initial cost and lowest cost are set to infinity.
'''


class Boid:
    def __init__(self, axis_limit, cost_function):
        self.position = [np.random.uniform(-axis_limit, axis_limit), np.random.uniform(-axis_limit, axis_limit)]
        self.velocity = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        self.cost = np.inf
        self.lowest_cost = np.inf
        self.best_position = self.position
        self.cost_function = cost_function
        self.axis_limit = axis_limit
        global best_cost_g
        global best_location_g

    '''
    New velocity function calculates the boids new velocity based on the old velocity, personal influence and social influence.
    Old velocity is modified by inertia to avoid velocity explosion.
    c1 = personal multiplier
    c2 = social multiplier
    c3 = global multiplier
    
    
    ADDED new line  + np.subtract(best_location_g, self.position
    '''

    def new_velocity(self, curr_iteration, swarms_best_location):
        # new personal velocity = old personal velocity + personal influence + social influence + social influence from all swarms
        c1 = 1
        c2 = 1
        c3 = 2
        inertia = 0.5
        self.velocity = np.multiply(inertia, self.velocity) + c1 * np.random.random() * (
            np.subtract(self.best_position, self.position)) + c2 * np.random.random() * (
                            np.subtract(swarms_best_location, self.position)) + c3 * np.random.random() * np.subtract(best_location_g, self.position)
        if np.abs(self.velocity[0]) > self.axis_limit/100 and np.abs(self.velocity[0]) > np.abs(self.velocity[1]):
            #print("if 1: ", self.velocity, np.abs(self.velocity[0]))
            self.velocity = self.velocity * ((self.axis_limit/100)/np.abs(self.velocity[0]))
            #print("if 1 post: ", self.velocity, np.abs(self.velocity[0]))
        elif np.abs(self.velocity[1]) > self.axis_limit/100:
            #print("if 2: ", self.velocity, np.abs(self.velocity[1]))
            self.velocity = self.velocity * ((self.axis_limit/100)/np.abs(self.velocity[1]))
            #print("if 2 post: ", self.velocity, np.abs(self.velocity[1]))


        return self.velocity

    '''
    New position is old position + velocity.
    The boid is kept within the axis limits if it would otherwise go out of bounds.
    '''

    def new_position(self, axis_limit):
        self.position = np.add(self.position, self.velocity)
        if self.position[0] > axis_limit:
            self.position[0] = axis_limit
        elif self.position[0] < -axis_limit:
            self.position[0] = -axis_limit
        if self.position[1] > axis_limit:
            self.position[1] = axis_limit
        elif self.position[1] < -axis_limit:
            self.position[1] = -axis_limit
        return self.position

    '''
    Update cost of the boid.
    If the new cost is best so far, update lowest cost and best position.
    '''

    def calculate_cost(self, cost_function, position):
        self.cost = cost_function(position)
        if self.cost < self.lowest_cost:
            self.lowest_cost = self.cost
            self.best_position = self.position
        return self.cost


'''
NEW STUFF USE ONLY IF WORKS
'''

class Swarm_of_swarms:
    def __init__(self, no_of_swarms, axis_limit, cost_function):
        self.axis_limit = axis_limit
        self.members = []
        self.no_of_swarms = no_of_swarms
        self.overall_best_cost = np.inf
        self.overall_best_location = [np.inf, np.inf]

'''
Swarm cosists of boids.
Swarm keeps swarms best position and cost in variables for the members to use in updating their velocities.
'''


class Swarm:
    def __init__(self, no_of_members, axis_limit, cost_function):
        self.axis_limit = axis_limit
        self.members = []
        self.no_of_members = no_of_members
        self.swarms_best_cost = np.inf
        self.swarms_best_location = [np.inf, np.inf]

        # xys variable is for plotting the data
        self.xs = []
        self.ys = []
        self.zs = []
        # create one list for each of the swarm members
        for i in range(no_of_members):
            self.xs.append(list())
            self.ys.append(list())
            self.zs.append(list())

        # create the members of the swarm
        for i in range(no_of_members):
            self.members.append(Boid(axis_limit=self.axis_limit, cost_function=cost_function))


'''
The optimizer runs the optimization algorithm by creating 5 swarms and using PSO to find the global minimum.
The optimizer also includes the animation functions. 
'''


class Optimizer:

    def __init__(self, swarm_size, cost_function, iterations, axis_limit, no_of_swarms):
        self.axis_limit = axis_limit
        self.cost_function = cost_function
        self.iterations = iterations
        self.swarms = dict()
        self.costs = dict()
        # Create 5 swarms and the lists of costs
        for i in range(no_of_swarms):
            self.swarms[i] = Swarm(swarm_size, axis_limit=self.axis_limit, cost_function=cost_function)
            self.costs[i] = [-1] * swarm_size
        global best_location_g
        global best_cost_g

        # global best costs and positions for plotting purposes
        self.gbcs = []
        self.gbps = []

    '''
    Optimize function runs the particle swarm optimization for as many iterations as is specified.
    The algorithm follow this framework:
        - for number of iterations
            1 - evaluate every boids fitness and update costs
            2 - find out each swarms best cost and position
            3 - find out globally best cost and position
            4 - update each boids velocity
            5 - update each boids location
            6 - return to to 1 if current iteration is less then specified
    '''

    def optimize(self):
        global best_location_g
        global best_cost_g
        for i in range(self.iterations):

            # evaluate every Boids fitness
            for k, swarm_name in enumerate(self.swarms):
                for j, _ in enumerate(self.swarms[swarm_name].members):
                    self.costs[swarm_name][j] = self.swarms[swarm_name].members[j].calculate_cost(self.cost_function,
                                                                                                  self.swarms[
                                                                                                      swarm_name].members[
                                                                                                      j].position)

                    # update swarms best cost and position
                    if self.costs[swarm_name][j] < self.swarms[swarm_name].swarms_best_cost:
                        self.swarms[swarm_name].swarms_best_cost = self.costs[swarm_name][j]
                        self.swarms[swarm_name].swarms_best_location = self.swarms[swarm_name].members[j].position
                    # update globally best cost and position
                    if self.costs[swarm_name][j] < best_cost_g:
                        best_cost_g = self.costs[swarm_name][j]
                        best_location_g = self.swarms[swarm_name].members[j].position

            # Recording best cost for plotting purposes
            self.gbcs.append(np.round(best_cost_g, 5))
            self.gbps.append(np.round(best_location_g, 5))
            # update every boids velocity and position
            for i, swarm_name in enumerate(self.swarms):
                for k, _ in enumerate(self.swarms[swarm_name].members):
                    # Here I record the Boid positions for plotting
                    self.swarms[swarm_name].xs[k].append(self.swarms[swarm_name].members[k].position[0])
                    self.swarms[swarm_name].ys[k].append(self.swarms[swarm_name].members[k].position[1])
                    self.swarms[swarm_name].zs[k].append(self.swarms[swarm_name].members[k].cost)

                    # algorithm continues
                    self.swarms[swarm_name].members[k].new_position(axis_limit=self.axis_limit)
                    self.swarms[swarm_name].members[k].new_velocity(curr_iteration=i + 1,
                                                                    swarms_best_location=self.swarms[
                                                                        swarm_name].swarms_best_location)

        return best_cost_g, best_location_g

    '''
    The Schwefel animation fuction creates a figure and 3d plot with a surface that follows the cost fuction.
    Each swarm has a different colors in the plot.
    The animation is saved as an mp4 file to the specified location locally.
    '''

    def animate_schwefel(self, title):
        from matplotlib import pyplot as plt
        from matplotlib import animation as ani
        plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\roope\Ohjelmat\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe'
        plt.rcParams['axes.titlesize'] = 'medium'


        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        figure, axis = plt.subplots(subplot_kw={"projection": "3d"})
        axis.set(xlim=[-self.axis_limit, self.axis_limit], ylim=[-self.axis_limit, self.axis_limit],
                 zlim=[0, 3 * self.axis_limit])
        surface_x = np.arange(-self.axis_limit, self.axis_limit + 1, 1)
        surface_y = np.arange(-self.axis_limit, self.axis_limit + 1, 1)
        surface_x, surface_y = np.meshgrid(surface_x, surface_y)
        axis.plot_surface(surface_x, surface_y, self.cost_function([surface_x, surface_y]), cmap="Blues", alpha=0.3)
        points_to_draw = []
        for _, swarm_name in enumerate(self.swarms):
            for i, _ in enumerate(self.swarms[swarm_name].xs):
                point = axis.scatter(self.swarms[swarm_name].xs[i][0], self.swarms[swarm_name].ys[i][0],
                                     self.swarms[swarm_name].zs[i][0], c=colors[swarm_name])
                points_to_draw.append(point)

        def update_schwefel(frame):
            plt.cla()
            plt.title(
                f'\n{title}, step: {frame + 1}\n Global best cost: {self.gbcs[frame]}\n Global best location: {self.gbps[frame]}')
            axis.set(xlim=[-self.axis_limit, self.axis_limit], ylim=[-self.axis_limit, self.axis_limit],
                     zlim=[0, 3 * self.axis_limit])
            axis.plot_surface(surface_x, surface_y, self.cost_function([surface_x, surface_y]), cmap="Blues", alpha=0.3)
            for _, swarm_name in enumerate(self.swarms):
                for j, _ in enumerate(self.swarms[swarm_name].xs):
                    x = self.swarms[swarm_name].xs[j][frame]
                    y = self.swarms[swarm_name].ys[j][frame]
                    z = self.swarms[swarm_name].zs[j][frame]
                    points_to_draw[j] = axis.scatter(x, y, z, c=colors[swarm_name])

        anim = ani.FuncAnimation(fig=figure, func=update_schwefel, frames=self.iterations, interval=500, repeat=False)
        anim.save(filename="pso_schwefel_video_file.mp4", fps=8)
        video = anim.to_html5_video()
        plt.close()
        return video

    '''

        The Banana animation fuction creates a figure and a 2d plot with a surface that follows the cost fuction.
        Each swarm has a different colors in the plot.
        The animation is saved as an mp4 file to the specified location locally.
        '''

    def animate_banana(self, title):
        from matplotlib import pyplot as plt
        from matplotlib import animation as ani
        plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\roope\Ohjelmat\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe'
        plt.rcParams['axes.titlesize'] = 'medium'


        colors = ['r', 'g', 'b', 'c', 'm']
        figure, axis = plt.subplots()
        axis.set(xlim=[-self.axis_limit, self.axis_limit], ylim=[-self.axis_limit, self.axis_limit])
        surface_x = np.arange(-self.axis_limit, self.axis_limit + 0.1, 0.1)
        surface_y = np.arange(-self.axis_limit, self.axis_limit + 0.1, 0.1)
        surface_x, surface_y = np.meshgrid(surface_x, surface_y)
        surface_color = axis.pcolor(self.cost_function([surface_x, surface_y]), vmax=300, alpha=0.3, cmap="RdBu")
        figure.colorbar(surface_color, ax=axis)
        points_to_draw = []
        for _, swarm_name in enumerate(self.swarms):
            for i, _ in enumerate(self.swarms[swarm_name].xs):
                point = axis.scatter(self.swarms[swarm_name].xs[i][0], self.swarms[swarm_name].ys[i][0],
                                     c=colors[swarm_name])
                points_to_draw.append(point)

        def update_banana(frame):
            plt.cla()
            plt.title(
                f'{title}, step: {frame + 1}\n Global best cost: {self.gbcs[frame]}, location: {self.gbps[frame]}')
            axis.set(xlim=[-self.axis_limit, self.axis_limit], ylim=[-self.axis_limit, self.axis_limit])
            # axis.plot_surface(surface_x, surface_y, self.cost_function([surface_x, surface_y]), cmap="Blues", alpha=0.3)
            surface_color = axis.pcolor(surface_x, surface_y, self.cost_function([surface_x, surface_y]), vmax=300,
                                        alpha=0.3, cmap="RdBu")
            for _, swarm_name in enumerate(self.swarms):
                for j, _ in enumerate(self.swarms[swarm_name].xs):
                    x = self.swarms[swarm_name].xs[j][frame]
                    y = self.swarms[swarm_name].ys[j][frame]
                    points_to_draw[j] = axis.scatter(x, y, c=colors[swarm_name])

        anim = ani.FuncAnimation(fig=figure, func=update_banana, frames=self.iterations, interval=500, repeat=False)
        anim.save(filename="../../tsp/static/videos/pso_banana_video_file.mp4", fps=8)
        video = anim.to_html5_video()
        plt.close()
        return video

if __name__ == "__main__":


    # Running of the optimizer and reporting the best cost and location found by the swarms.
    pso_schwefel = Optimizer(swarm_size=10, cost_function=schwefel, iterations=200, axis_limit=500, no_of_swarms=5)
    pso_schwefel_cost, pso_schwefel_location = pso_schwefel.optimize()
    print(f"Best Schwefel cost: {pso_schwefel_cost}, and location: {pso_schwefel_location}")

    # Reset best global cost and position for the second round of optimization
    best_cost_g = np.inf
    best_location_g = [np.inf, np.inf]

    # Running of the optimizer and reporting the best cost and location found by the swarms.
    pso_banana = Optimizer(swarm_size=10, cost_function=banana, iterations=200, axis_limit=2, no_of_swarms=5)
    pso_banana_cost, pso_banana_location = pso_banana.optimize()
    print(f"Best Banana cost: {pso_banana_cost}, and location: {pso_banana_location}")

    pso_schwefel.animate_schwefel("Schwefel plot")
    pso_banana.animate_banana("Banana plot")
