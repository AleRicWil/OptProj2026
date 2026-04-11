from location_generator import *
from node_machine import *
from simulated_annealing import *
import math
import numpy as np
import matplotlib.pyplot as plt

campus_plot = campus(100, (40.245751,-111.649794), (40.248344,-111.646590))
plot_map(campus_plot)

points = visitables(campus_plot)
campus_network = Network()
campus_network.add_points([Point(y, x) for y, x in points])
campus_network.plot_network()

# get a list of all points of interest

# generating points of interest at building corners

# look at clusters of 5 or 6 points of interest, and put a node near the centroid 
# (so long as it isn't in a building or something)


# draw lines from poi to centroid, and from centroid to centroid?