import matplotlib.pyplot as plt
import math
import random

def point_generator(n=50, l=100, w=100):
    '''
    gives us n points along x->l and y->w
    '''
    points: list[tuple[float, float]] = [(0, 0)]
    for _ in range(n-1):
        points.append((l*random.random(), w*random.random()))
    return points


def greedy(points):
    '''
    greedily reorders found points starting from first
    '''
    ordering = []
    others = points.copy()
    distance = 0

    # start point
    current = others.pop(0)
    ordering.append(current)

    while others:
        # find closest point from all others (iterate across all)
        nearest = min(others, key=lambda p: math.dist(current, p))
        distance += math.dist(current, nearest)

        # update and carry on
        ordering.append(nearest)
        others.remove(nearest)
        current = nearest
    
    distance += math.dist(current, ordering[0])

    return ordering, distance


def plot_path(points, distance):
    '''
    plots the path defined by an ordered list of (x, y) points
    '''

    x = [p[0] for p in points]
    y = [p[1] for p in points]

    plt.figure(figsize=(8, 6))

    # draw points
    plt.scatter(x, y, color='blue')

    # draw path lines and close loop
    plt.plot(x, y, color='red', linestyle='-', linewidth=1)
    plt.plot([x[-1], x[0]], [y[-1], y[0]], color='red', linewidth=1)

    plt.title(f"TSP Path, Distance = {distance}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


def path_distance(points):
    '''total distance of a closed TSP tour'''
    dist = 0
    for i in range(len(points) - 1):
        dist += math.dist(points[i], points[i+1])
    dist += math.dist(points[-1], points[0])  # close loop
    return dist


def neighbor(points):
    '''uses the 2-opt neighbor rule, which is supposedly much better'''
    new_points = points.copy()
    i, j = sorted(random.sample(range(len(points)), 2))
    new_points[i:j] = reversed(new_points[i:j])
    return new_points


def simulated_annealing(points, T=2000, alpha=0.999, kmax=40000):
    '''simulated annealing for TSP'''
    # build copies
    current = points.copy()
    current_dist = path_distance(current)
    best = current.copy()
    best_dist = current_dist
    distances = []

    for _ in range(kmax):
        new = neighbor(current)
        new_dist = path_distance(new)

        delta = new_dist - current_dist

        if delta <= 0:
            # always accept better solution
            current = new
            current_dist = new_dist
        else:
            # accept worse solution with probability
            r = random.random()
            P = math.exp(-delta / T)

            if P >= r:
                current = new
                current_dist = new_dist

        # track best solution found
        if current_dist < best_dist:
            best = current.copy()
            best_dist = current_dist

        # update
        T *= alpha
        distances.append(current_dist)

        # stop if temperature is very low
        if T < 1e-8:
            break

    return best, best_dist, distances

def plot_sa_convergence(distances):
    '''plots the evolution of the tour distance over iterations'''
    plt.figure(figsize=(10, 6))
    plt.plot(distances, color='blue', linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Tour Distance")
    plt.title("Simulated Annealing Convergence")
    plt.grid(True)
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    random.seed(4)
    points = point_generator()
    greedy_list, distance = greedy(points)
    print(f"Solution found with length: {distance}")
    plot_path(greedy_list, distance)

    sa_list, distance, distances = simulated_annealing(greedy_list)
    print(f"Solution found with length: {distance}")
    plot_path(sa_list, distance)
    plot_sa_convergence(distances)


