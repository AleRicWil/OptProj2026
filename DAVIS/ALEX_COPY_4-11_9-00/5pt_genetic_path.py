import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import random
import argparse
import heapq
from collections import defaultdict, deque
import math
from itertools import combinations

#####################################################
### MULTI-POI + USER-SELECTABLE POI PLACEMENT     ###
### (Random OR Regular Polygon) — GA-Ready!       ###
### + PATH-BIASED CARRY-OVER & MUTATION (NEW)     ###
#####################################################

# Material codes
material = {
    "open": 0,
    "paved": 1,
    "poi": 5
}

# Colors
color_list = [
    (0.9, 0.95, 0.9),   # 0: open grass
    (0.15, 0.15, 0.25), # 1: paved
    (0.9, 0.6, 0.7)     # 5: POI
]

def create_empty_grid(rows, cols):
    return np.zeros((rows, cols), dtype=np.uint8)

def place_random_pois(grid, num_points, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rows, cols = grid.shape
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(all_cells)
    poi_locations = all_cells[:num_points]
    for r, c in poi_locations:
        grid[r, c] = material["poi"]
    return grid, poi_locations

def place_polygon_pois(grid, num_points, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rows, cols = grid.shape
    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0
    radius = min(rows, cols) * 0.37
    poi_locations = []
    offset = -math.pi / 2
    for i in range(num_points):
        angle = offset + 2 * math.pi * i / num_points
        r = int(round(center_r + radius * math.sin(angle)))
        c = int(round(center_c + radius * math.cos(angle)))
        r = max(3, min(rows - 4, r))
        c = max(3, min(cols - 4, c))
        grid[r, c] = material["node"]
        poi_locations.append((r, c))
    return grid, poi_locations

def place_pois(grid, num_points, mode='polygon', seed=None):
    if mode == 'polygon':
        return place_polygon_pois(grid, num_points, seed)
    return place_random_pois(grid, num_points, seed)

def place_random_paved(grid, num_paved, seed=None):
    if seed is not None:
        random.seed(seed + 1)
        np.random.seed(seed + 1)
    rows, cols = grid.shape
    open_cells = [(r, c) for r in range(rows) for c in range(cols)
                  if grid[r, c] == material["open"]]
    random.shuffle(open_cells)
    actual_paved = min(num_paved, len(open_cells))
    paved_locations = open_cells[:actual_paved]
    for r, c in paved_locations:
        grid[r, c] = material["paved"]
    return grid, paved_locations

# ====================== GEOMETRIC A* (unchanged) ======================
def is_walkable(grid, r, c):
    rows, cols = grid.shape
    if 0 <= r < rows and 0 <= c < cols:
        return grid[r, c] in (material["paved"], material["poi"], material["door"], material["node"])
    return False

def get_neighbors_with_cost(grid, r, c):
    directions = [
        (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
        (0, -1, 1.0), (0, 1, 1.0),
        (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2))
    ]
    neighbors = []
    for dr, dc, cost in directions:
        nr, nc = r + dr, c + dc
        if is_walkable(grid, nr, nc):
            neighbors.append(((nr, nc), cost))
    return neighbors

def euclidean_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def find_shortest_path(grid, start, goal):
    if not is_walkable(grid, *start) or not is_walkable(grid, *goal):
        return None, None
    if start == goal:
        return [start], 0.0
    open_set = []
    counter = 0
    heapq.heappush(open_set, (euclidean_dist(start, goal), counter, start))
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0.0
    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            curr = current
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(start)
            path.reverse()
            return path, g_score[goal]
        for (neighbor, move_cost) in get_neighbors_with_cost(grid, *current):
            tentative_g = g_score[current] + move_cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + euclidean_dist(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))
    return None, None

def plot_grid(grid, title="Campus Walkway Grid", all_paths=None):
    cmap = ListedColormap(color_list)
    bounds = [-0.5, 0.5, 3.0, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(11, 9))
    plt.imshow(grid, cmap=cmap, norm=norm, origin="upper")
    plt.title(title)
    plt.xlabel("Columns (x-direction)")
    plt.ylabel("Rows (y-direction)")
    plt.gca().set_aspect('equal')
    if all_paths:
        for path in all_paths:
            if path and len(path) > 1:
                py = [r for r, c in path]
                px = [c for r, c in path]
                plt.plot(px, py, color='yellow', linewidth=2.5, alpha=0.65)
    legend_elements = [
        Patch(facecolor=color_list[0], edgecolor='black', label='Open Grass (free)'),
        Patch(facecolor=color_list[1], edgecolor='black', label='Paved Walkway'),
        Patch(facecolor=color_list[2], edgecolor='black', label='Point of Interest')
    ]
    if all_paths:
        legend_elements.append(Line2D([0], [0], color='yellow', lw=3, alpha=0.7, label='All Pairwise Paths'))
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1), frameon=True)
    plt.tight_layout()
    plt.show(block=True)

# ====================== NEW: PATH-BIASED GA HELPERS ======================
# Teaching note (Engineering Design Optimization, Ch. 7.6): 
# We now inject domain knowledge (the shortest-path "skeleton") into both 
# initialization and mutation. This follows the book's advice on using 
# problem structure to improve GA performance on combinatorial subset 
# selection problems.

def get_path_cells(grid, poi_list):
    """Returns the union of all cells that lie on any pairwise shortest path.
    Teaching: This is the 'connectivity skeleton'. Forcing these cells paved
    when moving to a tighter budget (next stage) guarantees we never lose
    connectivity — exactly the kind of building-block preservation the book
    discusses."""
    path_cells = set()
    for start, goal in combinations(poi_list, 2):
        path, _ = find_shortest_path(grid, start, goal)
        if path:
            path_cells.update(path)
    return path_cells

def compute_nearest_path_dist(rows, cols, path_cells):
    """Multi-source BFS distance to nearest path cell (O(grid) time).
    Teaching: Far more efficient than looping hypot() for every cell.
    Used to create exponential decay weights for biased selection."""
    dist = np.full((rows, cols), np.inf)
    if not path_cells:
        return dist
    q = deque(path_cells)
    for r, c in path_cells:
        dist[r, c] = 0
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and dist[nr, nc] > dist[r, c] + 1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist

def get_biased_weights(candidates, path_cells, rows, cols, bias_strength=5.0):
    """Returns weight list for weighted sampling (higher = nearer to paths).
    Teaching: Exponential decay gives strong preference near paths while
    still allowing exploration farther away."""
    dist_map = compute_nearest_path_dist(rows, cols, path_cells)
    weights = []
    for r, c in candidates:
        d = dist_map[r, c] if np.isfinite(dist_map[r, c]) else 50.0
        w = 1.0 + bias_strength * math.exp(-d / 3.0)
        weights.append(w)
    return weights

def create_path_biased_layout(base_grid, num_paved, path_cells, all_open_cells, rows, cols):
    """Teaching: Forces the entire previous-stage path skeleton to stay paved,
    then fills remaining slots with strong proximity bias.
    This is how we carry the 'best 10' paths into the next (smaller-budget) stage."""
    # Only keep paved cells from the path (POIs are already in base_grid)
    fixed_paved = {cell for cell in path_cells if base_grid[cell[0], cell[1]] == material["open"]}
    extra_needed = num_paved - len(fixed_paved)
    if extra_needed <= 0:
        return set(random.sample(list(fixed_paved), num_paved))
    open_cand = [c for c in all_open_cells if c not in fixed_paved]
    weights = get_biased_weights(open_cand, path_cells, rows, cols)
    probs = np.array(weights) / np.sum(weights)
    idx = np.random.choice(len(open_cand), size=extra_needed, replace=False, p=probs)
    extra = [open_cand[i] for i in idx]
    return fixed_paved.union(extra)

def biased_mutate(individual, num_paved, all_open_cells, base_grid, poi_list, rows, cols, mutation_rate=0.25):
    """MINIMAL CHANGE (your exact request) — far cells now prefer to become grass.
    Teaching (Martins & Ning, Ch. 7.6.2): Mutation is now an *informed* variation
    operator. We still add cells near paths (good for filling gaps), but we
    preferentially REMOVE (turn to grass) cells that are FAR from the current
    shortest paths. This prunes wasteful paved squares and forces the GA to
    "hone in" on the emerging optimal corridor network as the paved budget drops.
    """
    if random.random() >= mutation_rate:
        return individual.copy()
    
    # --- Recompute paths for this parent (same as before) ---
    temp_grid = base_grid.copy()
    for r, c in individual:
        temp_grid[r, c] = material["paved"]
    path_cells = get_path_cells(temp_grid, poi_list)
    
    # --- Reuse existing distance map (zero extra cost) ---
    dist_map = compute_nearest_path_dist(rows, cols, path_cells)
    
    ind_list = list(individual)
    open_candidates = [c for c in all_open_cells if c not in individual]
    
    num_flip = max(1, int(num_paved * 0.02))   # ~2% of paved cells flipped
    
    if not open_candidates or not ind_list:
        return individual.copy()
    
    # === 1. Addition still biased NEAR paths (unchanged logic) ===
    weights_add = []
    for r, c in open_candidates:
        d = dist_map[r, c] if np.isfinite(dist_map[r, c]) else 50.0
        w = 1.0 + 5.0 * math.exp(-d / 3.0)          # high weight = close
        weights_add.append(w)
    probs_add = np.array(weights_add) / np.sum(weights_add)
    add_idxs = np.random.choice(len(open_candidates), size=num_flip,
                                replace=False, p=probs_add)
    to_add = [open_candidates[i] for i in add_idxs]
    
    # === 2. REMOVAL NOW BIASED FAR FROM PATHS (the only new part) ===
    # Teaching: Probability of turning a paved cell to grass = proportional
    # to its distance from nearest path. Far cells get removed first → grass!
    weights_remove = []
    for r, c in ind_list:
        d = dist_map[r, c] if np.isfinite(dist_map[r, c]) else 50.0
        w = 1.0 + 5.0 * math.exp(d / 4.0)           # high weight = FAR (exp growth)
        weights_remove.append(w)
    probs_remove = np.array(weights_remove) / np.sum(weights_remove)
    
    # Weighted sample which paved cells to unpave (turn to grass)
    remove_idxs = np.random.choice(len(ind_list), size=min(num_flip, len(ind_list)),
                                   replace=False, p=probs_remove)
    to_remove = [ind_list[i] for i in remove_idxs]
    
    # === Apply changes (exactly same cardinality) ===
    for rem in to_remove:
        ind_list.remove(rem)
    for addd in to_add:
        ind_list.append(addd)
    
    return set(ind_list)

def fitness(paved_set, base_grid, poi_list):
    grid = base_grid.copy()
    for r, c in paved_set:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            grid[r, c] = material["paved"]
    total_score = 0.0
    for start, goal in combinations(poi_list, 2):
        path, distance = find_shortest_path(grid, start, goal)
        if path is None:
            return float('inf')
        total_score += distance
    return total_score

def crossover(parent1, parent2, num_paved):
    union = parent1.union(parent2)
    if len(union) < num_paved:
        return parent1.copy()
    return set(random.sample(list(union), num_paved))

# ====================== UPDATED GA STAGE ======================
def run_ga_stage(base_grid, poi_list, num_paved, pop_size=100, max_gens=1000, prev_top10=None):
    rows, cols = base_grid.shape
    all_open_cells = [(r, c) for r in range(rows) for c in range(cols)
                      if base_grid[r, c] == material["open"]]
    population = []
    # === PATH-BIASED CARRY-OVER OF BEST 10 FROM PREVIOUS STAGE ===
    if prev_top10:
        num_seeded = min(10, pop_size)
        for i in range(num_seeded):
            prev_layout = prev_top10[i % len(prev_top10)]
            temp_grid = base_grid.copy()
            for r, c in prev_layout:
                temp_grid[r, c] = material["paved"]
            path_cells = get_path_cells(temp_grid, poi_list)
            layout = create_path_biased_layout(base_grid, num_paved, path_cells, all_open_cells, rows, cols)
            population.append(layout)
    # Fill rest randomly (standard GA init)
    for _ in range(pop_size - len(population)):
        sample_size = min(num_paved, len(all_open_cells))
        population.append(set(random.sample(all_open_cells, sample_size)))
    best_per_gen_scores = []
    best_overall_score = float('inf')
    best_overall_layout = None
    for gen in range(max_gens):
        scored_pop = [(ind, fitness(ind, base_grid, poi_list)) for ind in population]
        scored_pop.sort(key=lambda x: x[1])
        current_best_score = scored_pop[0][1]
        best_per_gen_scores.append(current_best_score)
        print(f"  Gen {gen+1:3d} | Best score: {current_best_score:.2f}" +
              (" (DISCONNECTED)" if math.isinf(current_best_score) else ""))
        if current_best_score < best_overall_score:
            best_overall_score = current_best_score
            best_overall_layout = scored_pop[0][0].copy()
        if len(best_per_gen_scores) >= 10 and len(set(best_per_gen_scores[-10:])) == 1:
            print("   → Best score unchanged for 50 generations. Ending stage early.")
            break
        if math.isinf(current_best_score):
            print("   → No connected layout. Stopping GA.")
            break
        new_population = [ind.copy() for ind, _ in scored_pop[:10]]  # elitism
        parents = [ind for ind, _ in scored_pop[:pop_size // 2]]
        for _ in range(pop_size - 10):
            parent = random.choice(parents)
            child = biased_mutate(parent, num_paved, all_open_cells, base_grid, poi_list, rows, cols)
            new_population.append(child)
        population = new_population
    top10_layouts = [ind.copy() for ind, _ in scored_pop[:10]]
    return best_overall_layout, best_overall_score, best_per_gen_scores[-10:], top10_layouts

# ====================== MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-POI campus grid + All-Pairs Score + PATH-BIASED GA")
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument("--num_points", type=int, default=3)
    parser.add_argument("--seed", type=int, default=np.random.randint(0,99))
    parser.add_argument("--poi_mode", type=str, choices=['random', 'polygon'], default='polygon')
    args = parser.parse_args()
    print(f"=== MULTI-POI Walkway Grid + PATH-BIASED GA ===")
    print(f"Grid size : {args.rows} × {args.cols}")
    print(f"POIs      : {args.num_points} ({args.poi_mode})")
    print(f"Seed      : {args.seed}\n")
    grid = create_empty_grid(args.rows, args.cols)
    grid, poi_list = place_pois(grid, args.num_points, mode=args.poi_mode, seed=args.seed)
    print(f"POI locations: {poi_list}")
    base_grid = grid.copy()
    total_cells = args.rows * args.cols
    current_paved = total_cells - len(poi_list)
    reduction = max(1000, int(0.05 * current_paved))
    stage = 0
    prev_top10 = None
    print("\n=== STARTING STAGED GENETIC ALGORITHM OPTIMIZATION (with path bias) ===")
    while current_paved > 0:
        reduction = min(1000, int(0.05 * current_paved))
        stage += 1
        print(f"\n--- STAGE {stage} : Optimizing for {current_paved} paved cells ---")
        best_paved_set, best_score, last10_scores, new_top10 = run_ga_stage(
            base_grid, poi_list, current_paved, prev_top10=prev_top10
        )
        best_grid = base_grid.copy()
        for r, c in best_paved_set:
            best_grid[r, c] = material["paved"]
        all_paths = []
        for start, goal in combinations(poi_list, 2):
            path, _ = find_shortest_path(best_grid, start, goal)
            if path:
                all_paths.append(path)
        mode_str = "Polygon" if args.poi_mode == 'polygon' else "Random"
        title = f"STAGE {stage} — {args.rows}×{args.cols} — {current_paved} Paved | Score = {best_score:.1f}"
        plot_grid(best_grid, title=title, all_paths=all_paths)
        print("Last 10 generations best scores:")
        for i, s in enumerate(last10_scores, 1):
            status = "DISCONNECTED" if math.isinf(s) else f"{s:.2f}"
            print(f"  Gen -{10-i+1:2d}: {status}")
        prev_top10 = new_top10
        current_paved -= reduction
        if stage > 100 or math.isinf(best_score):
            print("best score was infeasible")
            break
    print("\n✅ Optimization complete!")
    print("Try: python 5pt_genetic_path.py --poi_mode polygon --num_points 5 --rows 80 --cols 80")