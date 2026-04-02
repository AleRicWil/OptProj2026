import math
import heapq
from location_generator import *

directions = [
    (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]

# directions = [
#     (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)
# ]

def neighbors(cell, shape):
    """Yield valid 8-दिशाएँ (direction) neighbors داخل (inside of) grid bounds"""
    r, c = cell
    rows, cols = shape

    for dr, dc, cost in  directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc), cost

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True

def terrain_cost(map, r, c):
    if map[r, c] == material["blocked"]:
        return float('inf')
    elif map[r, c] == material["interior"]:
        return 10
    return 1

def dijkstra_path_builder(map):
    queue = []
    # dist[cell] = (distance, poi_id)
    dist = {} 
    parent = {}
    pois = visitables(map)
    
    uf = UnionFind(len(pois))
    # Potential edges: (weight, cell_a, cell_b, poi_a, poi_b)
    edges = []

    # initialize Dijkstra from all POIs simultaneously
    for poi_id, cell in enumerate(pois):
        cell = tuple(cell)
        dist[cell] = (0, poi_id)
        parent[cell] = None
        heapq.heappush(queue, (0, cell, poi_id))

    # move in as straight of lines as you can until everyone is connected,
    # keeping in mind the costs of motion.
    while queue:
        d, cell, p_id = heapq.heappop(queue)

        if d > dist[cell][0]:
            continue

        for neighbor, move_cost in neighbors(cell, map.shape):
            nr, nc = neighbor
            if map[nr, nc] == material["blocked"]:
                continue
                
            new_dist = d + move_cost * terrain_cost(map, nr, nc)

            if neighbor not in dist:
                dist[neighbor] = (new_dist, p_id)
                parent[neighbor] = cell
                heapq.heappush(queue, (new_dist, neighbor, p_id))
            else:
                other_dist, other_id = dist[neighbor]
                if other_id != p_id:
                    # We found a bridge between two different POI regions!
                    # Total weight of this potential MST edge
                    total_weight = new_dist + other_dist
                    edges.append((total_weight, cell, neighbor, p_id, other_id))

    # Sort potential edges by weight (Kruskal's style)
    edges.sort(key=lambda x: x[0])

    path_cells = set()

    def get_full_path(c):
        curr = c
        while curr is not None:
            path_cells.add(curr)
            curr = parent[curr]

    for _, cell_a, cell_b, id_a, id_b in edges:
        if uf.union(id_a, id_b):
            # This edge connects two previously disconnected components
            get_full_path(cell_a)
            get_full_path(cell_b)

    # Apply to map
    for r, c in path_cells:
        if map[r, c] == material["open"]:
            map[r, c] = material["paved"]
            
def steiner_path_builder(map):
    """Near-Steiner path builder"""
    pois = [tuple(p) for p in visitables(map)]
    if not pois: return
    
    # 1. Start with one POI in the "Tree"
    connected_network = {pois[0]}
    remaining_pois = set(pois[1:])
    
    # Tracking the path cells
    all_path_cells = set()

    def get_terrain_cost(r, c):
        # Already paved? Very cheap to encourage Steiner junctions
        if (r, c) in all_path_cells or map[r, c] == material["door"]:
            return 0.001 
        if map[r, c] == material["blocked"]:
            return float('inf')
        return 1.0

    # 2. Iterate until all POIs are connected
    while remaining_pois:
        queue = []
        dist = {}
        parent = {}
        
        # Multi-source Dijkstra: Every cell currently in our tree is a starting point
        # This allows the next POI to connect to the CLOSEST point on the existing paths
        start_nodes = all_path_cells if all_path_cells else connected_network
        for cell in start_nodes:
            dist[cell] = 0
            heapq.heappush(queue, (0, cell))

        target_poi = None
        
        while queue:
            d, cell = heapq.heappop(queue)

            if cell in remaining_pois:
                target_poi = cell
                break

            if d > dist.get(cell, float('inf')):
                continue

            for neighbor, move_cost in neighbors(cell, map.shape):
                cost = get_terrain_cost(*neighbor)
                if cost == float('inf'): continue
                
                # move_cost is 1 for cardinal, 1.414 for diagonal
                new_dist = d + (move_cost * cost)

                if new_dist < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_dist
                    parent[neighbor] = cell
                    heapq.heappush(queue, (new_dist, neighbor))

        # 3. Add the newly found path to our network
        if target_poi:
            curr = target_poi
            while curr is not None and curr not in start_nodes:
                all_path_cells.add(curr)
                if map[curr[0], curr[1]] == material["open"]:
                    map[curr[0], curr[1]] = material["paved"]
                curr = parent.get(curr)
            
            remaining_pois.remove(target_poi)
            connected_network.add(target_poi)

if __name__=="__main__":
    # simple = simple_space(10, 10, "corridor")
    simple = simple_space(10, 10, "corners")
    print_map(simple)
    plot_map(simple)

    s1 = simple.copy()
    s2 = simple.copy()

    # find the paths
    dijkstra_path_builder(s1)
    print_map(s1)
    plot_map(s1)

    # find the paths
    steiner_path_builder(s2)
    print_map(s2)
    plot_map(s2)

    # again but bigger hey lets see what happens
    campus_plot = campus(128, (40.245751,-111.649794), (40.248344,-111.646590))
    plot_map(campus_plot)

    # find the paths
    c1 = campus_plot.copy()
    dijkstra_path_builder(c1)
    print_map(c1)
    plot_map(c1)

    # find the paths
    c2 = campus_plot.copy()
    steiner_path_builder(c2)
    print_map(c2)
    plot_map(c2)