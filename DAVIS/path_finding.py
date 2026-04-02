from location_generator import *
import math as m
import heapq


material = {
    "open": 0,
    "paved": 1,
    "door": 2,
    "blocked": 3,
    "interior": 4,
    "poi": 5,
    "node": 6
}


def bresenham_line(x0, y0, x1, y1):
    """return list of grid cells between two points (inclusive)."""
    cells = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            cells.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            cells.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    cells.append((x1, y1))
    return cells


def connect_local_pois(grid, max_distance):
    visited_pairs = set()
    pois = visitables(grid)

    for i, poi_a in enumerate(pois):
        for j, poi_b in enumerate(pois):
            if i >= j:
                continue    # avoid duplicates + self
            elif (i, j) in visited_pairs:
                continue    # avoid repeats
            elif m.dist(poi_a, poi_b) > max_distance:
                continue    # avoid out of bounds

            line_cells = bresenham_line(poi_a[0], poi_a[1], poi_b[0], poi_b[1])

            # handle blockages
            blocked = False
            for (x, y) in line_cells:
                if grid[x][y] == material["blocked"]:
                    blocked = True
                    break

            if blocked:
                continue  # avoid blocked connections

            # draw the path
            for (x, y) in line_cells:
                if grid[x][y] == material["open"]:
                    grid[x][y] = material["paved"]
            visited_pairs.add((i, j))

    return grid

def connect_nearest_pois(grid, k):
    visited_pairs = set()
    pois = visitables(grid)

    for i, poi_a in enumerate(pois):
        # compute distances to all other POIs
        distances = []
        for j, poi_b in enumerate(pois):
            if i == j:
                continue
            dist = m.dist(poi_a, poi_b)
            distances.append((dist, j, poi_b))

        # sort by distance
        distances.sort(key=lambda x: x[0])

        connections_made = 0

        for _, j, poi_b in distances:
            if connections_made >= k:
                break

            if (i, j) in visited_pairs:
                continue  # avoid duplicates

            line_cells = bresenham_line(poi_a[0], poi_a[1], poi_b[0], poi_b[1])

            # check for blockages
            blocked = False
            for (x, y) in line_cells:
                if grid[x][y] == material["blocked"]:
                    blocked = True
                    break

            if blocked:
                continue  # skip and try next nearest

            # draw the path
            for (x, y) in line_cells:
                if grid[x][y] == material["open"]:
                    grid[x][y] = material["paved"]

            visited_pairs.add((i, j))
            connections_made += 1

    return grid


def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])

    def heuristic(a, b):
        # Manhattan distance (good for grids)
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(x, y):
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                yield nx, ny, dx, dy

    def movement_cost(nx, ny, dx, dy):
        cell = grid[nx][ny]

        if dx != 0 and dy != 0:
            cost = m.sqrt(2)
        else:
            cost = 1.0

        if cell == material["paved"] or cell == material["interior"]:
            cost *= 0

        return cost

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for nx, ny, dx, dy in neighbors(*current):
            if grid[nx][ny] == material["blocked"]:
                continue

            tentative_g = g_score[current] + movement_cost(nx, ny, dx, dy)

            if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                came_from[(nx, ny)] = current
                g_score[(nx, ny)] = tentative_g
                f_score = tentative_g + heuristic((nx, ny), goal)
                heapq.heappush(open_set, (f_score, (nx, ny)))

    return None  # no path found


def connect_nearest_pois_astar(grid, k):
    visited_pairs = set()
    pois = visitables(grid)

    for i, poi_a in enumerate(pois):
        distances = []
        for j, poi_b in enumerate(pois):
            if i == j:
                continue
            dist = m.dist(poi_a, poi_b)
            distances.append((dist, j, poi_b))

        distances.sort(key=lambda x: x[0])

        connections_made = 0

        for _, j, poi_b in distances:
            if connections_made >= k:
                break

            if (i, j) in visited_pairs or (j, i) in visited_pairs:
                continue

            path = astar(grid, tuple(poi_a), tuple(poi_b))

            if path is None:
                continue  # no valid route

            # draw the path
            for (x, y) in path:
                if grid[x][y] == material["open"]:
                    grid[x][y] = material["paved"]

            visited_pairs.add((i, j))
            connections_made += 1

    return grid



def node_finder(grid):

    # across all cells in grid


        # place node just outside of doors
            # you could find these with a kernel, looking for a door flanked by walls, with interior on one side

        # place node at building corners
            # you could find these with a kernel, looking for an L-shape of 

        # place node at centroid of local door areas


    return grid



if __name__ == "__main__":
    # simple space
    simple = simple_space(10, 20, "corners")
    plot_map(simple)

    # connect all points along straight lines
    simple = connect_local_pois(simple, max_distance=30)
    plot_map(simple)

    # use the astar on the simple
    smart_simple = connect_nearest_pois_astar(simple.copy(), k=2)
    plot_map(smart_simple)

    # campus map
    campus_plot = campus(256, (40.245751,-111.649794), (40.248344,-111.646590))
    plot_map(campus_plot)

    # use k nearest neighbors on campus map
    campus_lines = connect_nearest_pois(campus_plot.copy(), k=1)
    plot_map(campus_lines)

    # connect literally all points along straight lines
    many_lines = connect_local_pois(campus_plot.copy(), max_distance=256)
    plot_map(many_lines)

    # connect all points to 1 nearest neighbors intelligently
    smart_lines = connect_nearest_pois_astar(campus_plot.copy(), k=1)
    plot_map(smart_lines)

    # connect all points to 4 nearest neighbors intelligently
    smart_lines = connect_nearest_pois_astar(campus_plot.copy(), k=4)
    plot_map(smart_lines)