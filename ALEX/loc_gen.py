# =============================================================================
# location_generator.py
# Helper added at the end
# for line-crossing checks against the raster map. This ensures paved paths never
# cross buildings when we build the initial greedy network.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#####################################################
### NOTE: COORDS ARE IN (LAT, LON) MEANING (Y, X) ###
### THAT IS HOW MATRICES HANDLE IT, AND ALSO MAPS ###
#####################################################

# options for materials. If you add one, add it to the colormap too
material = {
    "open": 0,
    "paved": 1,
    "door": 2,
    "blocked": 3,
    "interior": 4,
    "poi": 5,
    "node": 6,
    "destination": 7
}

color_map = {
    0: (0.9, 0.95, 0.9),    # open
    1: (0.7, 0.7, 0.8),     # paved
    2: (0.9, 0.5, 0.3),     # door
    3: (0.1, 0.2, 0.3),     # blocked
    4: (0.6, 0.6, 0.7),     # interior
    5: (0.5, 0.6, 0.7),      # point of interest (staircase, quad)
    6: (1,1,1),             # node, connections walkways in open space
    7: (0, 0, 0)            # destination
}

# https://www.latlong.net/
# try to keep the bottom left corner first and top right second. It doesn't matter for the code,
# but it can help with making sure the doors are properly aligned with the buildings
buildings = {
    "eb": ((40.246081,-111.648444), (40.246474,-111.647306)),
    "eb-tunnel": ((40.246355,-111.648266), (40.246862,-111.648201)),
    "clyde1": ((40.246591,-111.648354), (40.247315,-111.647668)),
    "clyde2": ((40.246698,-111.648447),(40.247111,-111.647831)),
    "marb": ((40.246622,-111.649448), (40.247032,-111.648960)),
    "kennedy1": ((40.247321,-111.649504), (40.247747,-111.649274)),
    "kennedy2": ((40.247673,-111.649504), (40.247850,-111.648971)),
    "bike racks": ((40.247416,-111.648414), (40.247878,-111.648414)),
    "wilk1": ((40.248023,-111.648414),(40.248707,-111.647110)),
    "wilk2": ((40.248236,-111.647775),(40.249063,-111.646353)),
    "wilk parking": ((40.246840,-111.647523), (40.247830,-111.64706195846482)),
    "hbll1": ((40.248073,-111.649679),(40.248422,-111.648825)),
    "hbll2": ((40.248229,-111.649385),(40.249363,-111.649116)),
    "hbll3": ((40.248593,-111.649754),(40.248999,-111.648739)),
}

# bridges from interiors to exteriors. Make CERTAIN these fall on the exact lines of the exteriors they touch
doors = {
    "eb_clyde": [(40.246253,-111.648444), (40.247062,-111.648447), (40.246739,-111.648447), (40.247315,-111.648222), (40.247076492202325, -111.647668),
                 (40.24607614438181, -111.64740418766316)],
    "marb": [(40.246622,-111.649200), (40.246830,-111.648960), (40.247032,-111.649200), 
             (40.246830,-111.649448)],
    "kennedy": [(40.247321,-111.649395), (40.247573,-111.649504), (40.247850,-111.649400), 
                (40.247850,-111.649094), (40.247580,-111.649274)],
    "wilk": [(40.248062,-111.648414), (40.248023,-111.647389), (40.248833,-111.647775), 
             (40.248707,-111.648289)],
    "parking_sidewalk_points": [(40.247830, -111.647523), (40.2473941084455, -111.647523), (40.247432301089816, -111.64706195846482), (40.247830, -111.64706195846482)],
    "library": [(40.248073,-111.649249), (40.249227,-111.649385), (40.249227,-111.649116)]
}

destinations = {
    "eb_lobby": [(40.246258312800094, -111.64832465274255)], "clyde_stepdown": [(40.24691410537213, -111.64832904495513)], "crab_front": [(40.247728397580175, -111.64686707014934)],
    "wilk_food": [(40.248168970669624, -111.64746378025232)], "wilk_store": [(40.24812068595096, -111.64822978028613)],
    "lsb_top": [(40.2459, -111.64923449161479)], "marb_central": [(40.24684, -111.64921911077138)], "kennedy_central": [(40.24761351405376, -111.64938765785497)],
    "library_south": [(40.24814, -111.64912712491088)], "little_study_zone": [(40.247759,-111.648712)], "bus_stop": [(40.245961210137274, -111.64681664610747)],
    "parking_central": [(40.24762641479833, -111.64734799544142)]
}


def simple_space(y, x, mode):
    """Generates a rectangular space with doors in the corners."""
    array = np.zeros((y, x), dtype=np.uint8)
    # doors in the corners
    if (mode == "corners"):
        array[0, 0] = material["door"]
        array[y-1, 0] = material["door"]
        array[0, x-1] = material["door"]
        array[y-1, x-1] = material["door"]
    elif (mode == "corridor"):
        array[y//2, 0] = material["door"]
        array[y//2, x-1] = material["door"]

    return array


def building(map, p1, p2):
    """Alters the incoming map by adding a blocked zone between the specified points."""
    y1, x1 = p1
    y2, x2 = p2

    miny = np.minimum(y1, y2) - 0.5
    maxy = np.maximum(y1, y2) + 0.5
    minx = np.minimum(x1, x2) - 0.5
    maxx = np.maximum(x1, x2) + 0.5

    w = maxy - miny
    h = maxx - minx
    # a building, bottom-left is (miny, minx)
    r, c = np.indices(map.shape)
    # only put down a building where there is not already one
    mask1 = (np.abs(r - (w/2 + miny)) < w/2) & (np.abs(c - (h/2 + minx)) < h/2) & (map == material["open"])
    map[mask1] = material["blocked"]

    # fill the interiors of buildings with walkable spaces
    mask2 = (np.abs(r - (w/2 + miny)) < w/2 - 1) & (np.abs(c - (h/2 + minx)) < h/2 - 1)
    map[mask2] = material["interior"]

    return map


def cost(map, units):
    """Returns the total number of paved spaces in the map, multiplied by the cost per paved square"""
    return np.count_nonzero(map == material["paved"])*units    


def print_map(map):
    """Prints the map out in plain text, for the small ones"""
    print(np.flip(map, 0))


def plot_map(map):
    """Plots the map nice and pretty with colors and such"""
    cmap = ListedColormap([color_map[i] for i in sorted(color_map.keys())])

    plt.figure()
    plt.imshow(map, cmap=cmap, origin="lower", vmin=0, vmax=len(color_map)-1)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.show()


def campus(resolution, p1, p2):
    """Builds a map of campus between two coordinates"""
    y1, x1 = p1
    y2, x2 = p2

    miny = np.minimum(y1, y2)
    maxy = np.maximum(y1, y2)
    minx = np.minimum(x1, x2)
    maxx = np.maximum(x1, x2)

    w = maxy - miny
    h = maxx - minx

    # convert from lat/long to scaled, integer coordinates
    mind = np.minimum(w, h)
    scale = int(resolution/mind)
    w = int(w*scale)
    h = int(h*scale)

    map = np.zeros((w, h), dtype=np.uint8)
    buildings_list = []

    for c1, c2 in buildings.values():
        point1 = (int(scale*(c1[0] - miny)), int(scale*(c1[1] - minx)))
        point2 = (int(scale*(c2[0] - miny)), int(scale*(c2[1] - minx)))
        buildings_list.append((point1, point2))
        map = building(map, point1, point2)

    for door_list in doors.values():
        for d in door_list:
            cy = int(scale * (d[0] - miny))   # center y (row)
            cx = int(scale * (d[1] - minx))   # center x (column)

            if not (0 <= cy < map.shape[0] and 0 <= cx < map.shape[1]):
                continue

            # Center cell
            map[cy, cx] = material["door"]

            # Four adjacent cells (up, down, left, right) — only if inside map
            # We only overwrite open/interior cells; never touch blocked walls
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < map.shape[0] and 0 <= nx < map.shape[1] and
                    map[ny, nx] in (material["open"], material["interior"])):
                    map[ny, nx] = material["door"]

    for dest_list in destinations.values():
        for coord in dest_list:
            y = int(scale*(coord[0] - miny))
            x = int(scale*(coord[1] - minx))

            # don't plot if it's index is negative
            if 0 <= y < map.shape[0] and 0 <= x < map.shape[1]:
                map[y, x] = material["destination"]
            else: print('oopse')

    return map, buildings_list


def visitables(map, mat="all"):
    """Returns a list of points that are be visitable"""
    if mat == "all":
        # special case for when we want all of em
        spots = np.argwhere(np.isin(map, [material["door"], material["poi"], material["node"]]))
    else:
        spots = np.argwhere(np.isin(map, [material[mat]]))

    return spots


# -----------------------------------------------------------------------------
# NEW HELPER (added for the walkway optimizer)
# -----------------------------------------------------------------------------
def line_crosses_building(p1_y: float, p1_x: float, p2_y: float, p2_x: float, campus_map) -> bool:
    """Checks if a straight-line path would cross a blocked building cell.
    Uses dense Euclidean sampling along the line (NOT grid-step counting).
    This is the exact distance function you asked for - pure geometry."""
    dy = p2_y - p1_y
    dx = p2_x - p1_x
    dist = (dy**2 + dx**2)**0.5
    if dist < 1e-6:
        return False
    n_samples = max(10, int(dist * 3))  # more samples = more accurate crossing detection
    for i in range(1, n_samples):
        t = i / n_samples
        y = int(round(p1_y * (1 - t) + p2_y * t))
        x = int(round(p1_x * (1 - t) + p2_x * t))
        if 0 <= y < campus_map.shape[0] and 0 <= x < campus_map.shape[1]:
            if campus_map[y, x] in [material["blocked"], material["interior"]]:
                return True
    return False


if __name__ == "__main__":
    campus_plot, _ = campus(100, (40.245751,-111.649794), (40.248344,-111.646590))
    plot_map(campus_plot)