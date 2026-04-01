from location_generator import *
import numpy as np
from math import sqrt


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


def is_blocked(grid, x, y, blocked_value):
    return grid[x][y] == blocked_value


def distance(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def connect_pois(grid, max_distance, material):
    visited_pairs = set()
    pois = visitables(grid)

    for i, poi_a in enumerate(pois):
        for j, poi_b in enumerate(pois):
            if i >= j:
                continue    # avoid duplicates + self

            if (i, j) in visited_pairs:
                continue    # avoid repeats

            if distance(poi_a, poi_b) > max_distance:
                continue    # avoid out of bounds

            line_cells = bresenham_line(poi_a[0], poi_a[1], poi_b[0], poi_b[1])

            # handle blockages
            blocked = False
            for (x, y) in line_cells:
                if is_blocked(grid, x, y, material["blocked"]):
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


simple = simple_space(10, 20, "corners")
plot_map(simple)

simple = connect_pois(simple, max_distance=30, material=material)
plot_map(simple)

campus_plot = campus(256, (40.245751,-111.649794), (40.248344,-111.646590))
plot_map(campus_plot)

campus_plot = connect_pois(campus_plot, max_distance=200, material=material)
plot_map(campus_plot)