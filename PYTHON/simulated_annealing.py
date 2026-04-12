import numpy as np
import matplotlib.pyplot as plt
from node_machine import *
from location_generator import cost_map

def tsp_length(tour, dist_matrix):
    """Calculate the total distance of a tour (list of city indices).
    The tour must start and end at the same city (usually city 0).
    This is the objective function we want to *minimize*."""
    total = 0.0
    for i in range(len(tour) - 1):
        total += dist_matrix[tour[i], tour[i + 1]]
    # Close the loop back to the starting city
    total += dist_matrix[tour[-1], tour[0]]
    return total

def networking_length(network: Network, k=3, cost_map=cost_map):
    # for each door in network
        # pick k other points in the network at random (random? we may want a better way to do this)
            # find the greedy shortest paths to each of these, individually, and average the lengths
                # if ever you CANNOT get to the other location, throw an error. Or return infinity?
                # record the longest path found
    
    # take the sum, and divide it by (num_doors) to get our average path length for the whole network
        # note, if we do this too randomly, this value can deviate a lot across evaluations. We may want a way
        # to take like, the nearest point, the farthest point, and k random points in the middle? 
    # return the length. Or something like that.

    return ...

def networking_angles(network: Network):

    # perform a similar search to networking_length(), but instead of using distances, record the angles required for action
    # at each intersection using Point.get_turn_angle(Path1, Path2).
    # i say, don't count any angles below 45 degrees. That may just be me though. We could return 2 * (radians**2) or
    # something like that to more aggressively reduce values beyond ~30 degrees or something like that

    return ...


def greedy_nearest_neighbor(dist_matrix, start_city=0):
    """Implements the greedy nearest-neighbor TSP heuristic.
    Returns a list of city indices forming a closed tour."""
    n = dist_matrix.shape[0]
    visited = [False] * n
    tour = [start_city]
    visited[start_city] = True
    current = start_city
    
    for _ in range(n - 1):
        # Find the closest unvisited city
        min_dist = np.inf
        next_city = -1
        for city in range(n):
            if not visited[city]:
                if dist_matrix[current, city] < min_dist:
                    min_dist = dist_matrix[current, city]
                    next_city = city
        # Move there
        tour.append(next_city)
        visited[next_city] = True
        current = next_city
    
    # Close the tour back to start (as required by the homework)
    tour.append(start_city)
    return tour


def simulated_annealing(dist_matrix, initial_tour, points, max_iter=20000,
                        initial_temp=10000.0, cooling_rate=0.9999):
    """Custom Simulated Annealing for TSP - NOW WITH EXPLICIT WORSE-PATH ACCEPTANCE.
    This is the exact edit you asked for: the algorithm will occasionally accept 
    a LONGER (worse) tour when temperature is high. This is what allows it to 
    'jump over a peak to a deeper valley' and escape local minima."""
    n = dist_matrix.shape[0]
    
    # Work with a copy of the tour (remove closing city for easier swapping)
    current_tour = initial_tour[:-1].copy()
    current_distance = tsp_length(current_tour + [current_tour[0]], dist_matrix)
    
    best_tour = current_tour[:]
    best_distance = current_distance
    
    temperature = initial_temp
    convergence_history = [best_distance]
    
    worse_accepted = 0          # NEW: track how many times we accept a longer path

    # --- Live path plot setup (runs once at start of SA) ---
    plt.ion()                                      # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 6))       
    ax.set_title('TSP - Live SA Path Evolution')
    ax.set_xlabel('x (units)')
    ax.set_ylabel('y (units)')
    ax.grid(True)
    
    # Plot static cities & start once
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=50, label='Cities')
    ax.scatter([points[0, 0]], [points[0, 1]], c='red', s=200, marker='*', label='Start')
    line_path, = ax.plot([], [], 'g-', linewidth=2, label='Current Best Path')
    ax.legend()
    plt.show(block=False)                          # Show figure without blocking
    
    print("\nStarting Simulated Annealing (with worse-path acceptance enabled)...")
    print(f"Initial temperature: {temperature:.0f} | Initial distance: {current_distance:.2f}")
    print("   (Watch for 'Accepted WORSE move' prints - this is the magic of SA!)\n")
    
    for iteration in range(max_iter):
        # Generate neighbor: random swap of any two cities
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        while i == j:
            j = np.random.randint(0, n)
        
        new_tour = current_tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        
        new_distance = tsp_length(new_tour + [new_tour[0]], dist_matrix)
        delta = new_distance - current_distance
        
        # =====================================================================
        # METROPOLIS ACCEPTANCE CRITERION - THIS IS THE EDIT YOU WANTED
        # =====================================================================
        #   • If delta < 0 → always accept (better path)
        #   • Else → accept with probability exp(-delta / T) → occasionally
        #     accepts a LONGER path when T is high
        # =====================================================================
        accepted = False
        if delta < 0:
            accepted = True
        else:
            prob = np.exp(-delta / temperature)
            if np.random.rand() < prob:
                accepted = True
                worse_accepted += 1
        
        if accepted:
            current_tour = new_tour
            current_distance = new_distance
        
        # Update best solution seen so far
        if current_distance < best_distance:
            best_tour = current_tour[:]
            best_distance = current_distance

        # --- Live update of the SAME figure every 100 iterations ---
        if iteration % 1000 == 0:
            # Build closed best tour for plotting
            best_closed = best_tour + [best_tour[0]]
            x = points[best_closed, 0]
            y = points[best_closed, 1]
            line_path.set_data(x, y)               # Redraw path on SAME figure
            ax.set_title(f'TSP - Live SA Path (iter {iteration}, dist {best_distance:.2f})')
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)                       # Tiny pause lets plot refresh
        
        convergence_history.append(best_distance)
        
        # Cool the system
        temperature *= cooling_rate
        
        if temperature < 1e-12:
            break
    
    # Close the best tour for plotting
    best_tour_closed = best_tour + [best_tour[0]]
    
    print(f"\nFinished SA after {len(convergence_history)-1} iterations.")
    print(f"Final SA distance: {best_distance:.2f}")
    print(f"Final temperature: {temperature:.4e}")
    plt.ioff()
    
    return best_tour_closed, best_distance, convergence_history


def homework_example():
    # Set random seed for reproducibility (so your plots match the ones you turn in)
    np.random.seed(4)

    n_cities = 50
    points = np.zeros((n_cities, 2))
    points[0] = [0.0, 0.0]                              # fixed starting point
    points[1:] = np.random.uniform(0, 100, size=(49, 2))  # 49 random points

    print("Points generated. Start city (index 0) is at (0,0).")
    print(f"Example random point (city 1): {points[1]}")

    # For TSP efficiency we pre-compute all pairwise Euclidean distances once.
    # This turns the problem into a lookup table (common in real routing apps).
    dist_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            # Euclidean distance between city i and city j
            dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])

    print("Distance matrix ready (50 x 50).")

    # Run greedy
    greedy_tour = greedy_nearest_neighbor(dist_matrix)
    greedy_distance = tsp_length(greedy_tour, dist_matrix)

    print("\n=== GREEDY SOLUTION ===")
    print(f"Greedy tour distance: {greedy_distance:.2f} units")

    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=50, label='Cities')
    plt.scatter([points[0, 0]], [points[0, 1]], c='red', s=200, marker='*', label='Start (0,0)')
    # Draw the path (including return to start)
    for i in range(len(greedy_tour) - 1):
        city_a = greedy_tour[i]
        city_b = greedy_tour[i + 1]
        plt.plot([points[city_a, 0], points[city_b, 0]],
                [points[city_a, 1], points[city_b, 1]], 'r-', linewidth=1.5)
    plt.title('TSP - Greedy Nearest-Neighbor Path')
    plt.xlabel('x (units)')
    plt.ylabel('y (units)')
    plt.grid(True)
    plt.legend()

    # Generate random initial tour (random layout)
    cities_to_shuffle = list(range(1, n_cities))      # exclude fixed start city 0
    np.random.shuffle(cities_to_shuffle)              # random permutation
    random_tour = [0] + cities_to_shuffle + [0]      # force start/end at city 0

    print("\n=== RANDOM INITIAL TOUR GENERATED ===")
    print(f"Random tour distance (before SA): {tsp_length(random_tour, dist_matrix):.2f}")

    # Run simulated annealing
    sa_tour, sa_distance, convergence = simulated_annealing(
        dist_matrix, random_tour, points, max_iter=2000000
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=50, label='Cities')
    plt.scatter([points[0, 0]], [points[0, 1]], c='red', s=200, marker='*', label='Start (0,0)')
    for i in range(len(sa_tour) - 1):
        city_a = sa_tour[i]
        city_b = sa_tour[i + 1]
        plt.plot([points[city_a, 0], points[city_b, 0]],
                [points[city_a, 1], points[city_b, 1]], 'g-', linewidth=1.5)
    plt.title('TSP - Improved Path after Simulated Annealing')
    plt.xlabel('x (units)')
    plt.ylabel('y (units)')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(convergence, color='purple', linewidth=1.5)
    plt.title('Convergence Plot - Best TSP Total Length vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Best Tour Length (units)')
    plt.grid(True)
    plt.show()

    print("\n=== FINAL RESULTS ===")
    print(f"Greedy algorithm total distance: {greedy_distance:.2f}")
    print(f"Simulated Annealing total distance: {sa_distance:.2f}")
    print(f"Improvement achieved: {((greedy_distance - sa_distance)/greedy_distance)*100:.1f}%")


if __name__ == "__main__":
    homework_example()
