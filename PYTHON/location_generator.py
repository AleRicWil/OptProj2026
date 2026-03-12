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
    "poi": 5
}

color_map = {
    0: (0.9, 0.95, 0.9),    # open
    1: (0.7, 0.7, 0.8),     # paved
    2: (0.9, 0.5, 0.3),     # door
    3: (0.1, 0.2, 0.3),     # blocked
    4: (0.6, 0.6, 0.7),     # interior
    5: (0.9, 0.6, 0.7)      # point of interest (staircase, quad)
}

# https://www.latlong.net/
# try to keep the bottom left corner first and top right second. It doesn't matter for the code,
# but it can help with making sure the doors are properly aligned with the buildings
buildings = {
    "eb": ((40.246081,-111.648444), (40.246474,-111.647306)),
    "clyde1": ((40.246591,-111.648354), (40.247315,-111.647668)),
    "clyde2": ((40.246698,-111.648447),(40.247111,-111.647831)),
    "marb": ((40.246622,-111.649448), (40.247032,-111.648960)),
    "kennedy1": ((40.247321,-111.649504), (40.247747,-111.649274)),
    "kennedy2": ((40.247673,-111.649504), (40.247850,-111.648971)),
    "bike racks": ((40.247416,-111.648414), (40.247878,-111.648414)),
    "wilk1": ((40.248023,-111.648414),(40.248707,-111.647110)),
    "wilk2": ((40.248236,-111.647775),(40.249063,-111.646353)),
    "wilk parking": ((40.246840,-111.647523), (40.247830,-111.64656)),
    "hbll1": ((40.248073,-111.649679),(40.248422,-111.648825)),
    "hbll2": ((40.248229,-111.649385),(40.249363,-111.649116)),
    "hbll3": ((40.248593,-111.649754),(40.248999,-111.648739)),
}

doors = {
    "eb": [(40.246253,-111.648444), (40.246474,-111.648222)],
    "clyde": [(40.247062,-111.648447), (40.246739,-111.648447), (40.247315,-111.648222), 
              (40.246591,-111.648222)],
    "marb": [(40.246622,-111.649200), (40.246830,-111.648960), (40.247032,-111.649200), 
             (40.246830,-111.649448)],
    "kennedy": [(40.247321,-111.649395), (40.247573,-111.649504), (40.247850,-111.649400), 
                (40.247850,-111.649094), (40.247580,-111.649274)],
    "wilk": [(40.248062,-111.648414), (40.248023,-111.647393), (40.248833,-111.647775), 
             (40.248707,-111.648289)],
    "library": [(40.248073,-111.649249), (40.249227,-111.649385), (40.249227,-111.649116)]
}

poi = {
    "south street": [(40.245824,-111.649303), (40.245877,-111.649180), 
                     (40.245992,-111.648810), (40.246008,-111.648579), (40.245957,-111.647407)],
    "central strip": [(40.247373,-111.648593), (40.247971,-111.648609), (40.247971,-111.648838), 
                      (40.247971,-111.649251), (40.247971,-111.649999)],
    "wilk parking sidewalk points": [(40.247830,-111.647389), (40.247391,-111.647523)],
}

"""Generates a rectangular space with doors in the corners."""
def simple_space(y, x, mode):
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

"""Alters the incoming array by adding a blocked zone between the specified points."""
def building(array, p1, p2):
    y1, x1 = p1
    y2, x2 = p2

    miny = np.minimum(y1, y2) - 0.5
    maxy = np.maximum(y1, y2) + 0.5
    minx = np.minimum(x1, x2) - 0.5
    maxx = np.maximum(x1, x2) + 0.5

    w = maxy - miny
    h = maxx - minx
    # a building, bottom-left is (miny, minx)
    r, c = np.indices(array.shape)
    # only put down a building where there is not already one
    mask1 = (np.abs(r - (w/2 + miny)) < w/2) & (np.abs(c - (h/2 + minx)) < h/2) & (array == material["open"])
    array[mask1] = material["blocked"]

    # fill the interiors of buildings with walkable spaces
    mask2 = (np.abs(r - (w/2 + miny)) < w/2 - 1) & (np.abs(c - (h/2 + minx)) < h/2 - 1)
    array[mask2] = material["interior"]

    return array

"""Returns the total number of paved spaces in the array, multiplied by the cost per paved square"""
def cost(array, units):
    return np.count_nonzero(array == material["paved"])*units    

"""Prints the array out, for small ones"""
def print_array(array):
    print(np.flip(array, 0))

"""Prints the grid nice and pretty"""
def plot_grid(grid):
    cmap = ListedColormap([color_map[i] for i in sorted(color_map.keys())])

    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap=cmap, origin="lower", vmin=0, vmax=len(color_map)-1)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.show()

"""Builds a map of campus between two coordinates"""
def campus(res, p1, p2):
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
    scale = int(res/mind)
    w = int(w*scale)
    h = int(h*scale)

    array = np.zeros((w, h), dtype=np.uint8)

    for c1, c2 in buildings.values():
        point1 = (int(scale*(c1[0] - miny)), int(scale*(c1[1] - minx)))
        point2 = (int(scale*(c2[0] - miny)), int(scale*(c2[1] - minx)))
        array = building(array, point1, point2)

    for door_list in doors.values():
        for d in door_list:
            y = int(scale*(d[0] - miny))
            x = int(scale*(d[1] - minx))

            # don't plot a door if it's index is negative
            if 0 <= y < array.shape[0] and 0 <= x < array.shape[1]:
                array[y, x] = material["door"]

    for poi_list in poi.values():
        for p in poi_list:
            y = int(scale*(p[0] - miny))
            x = int(scale*(p[1] - minx))

            # don't plot a door if it's index is negative
            if 0 <= y < array.shape[0] and 0 <= x < array.shape[1]:
                array[y, x] = material["poi"]

    return array

if __name__ == "__main__":
    # basic = simple_space(6, 6, "corners")
    # print_array(basic)
    # print(cost(basic, 10))
    # plot_grid(basic)

    # basic = simple_space(5, 5, "corridor")
    # plot_grid(basic)

    campus_plot = campus(72, (40.245462,-111.649794), (40.248344,-111.646590))
    plot_grid(campus_plot)