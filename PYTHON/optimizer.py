from location_generator import *
from node_machine import *
from simulated_annealing import *

# pull in our plot of campus
campus_plot, buildings_list = campus(20, (40.245751,-111.649794), (40.248344,-111.646590))
plot_map(campus_plot)

# get a list of all points of interest
doors = visitables(campus_plot, "door")
pois = visitables(campus_plot, "poi")

campus_network = Network()
campus_network.add_points([Point(y, x) for y, x in doors], material["door"])
campus_network.add_points([Point(y, x) for y, x in pois], material["poi"])

[campus_network.add_building(Point(y1, x1), Point(y2, x2)) for ((y1, x1), (y2, x2)) in buildings_list]

campus_network.plot_network()

# look at clusters of 5 or 6 points of interest, and put a node near the centroid 
# (so long as it isn't in a building or something)


# draw lines from poi to centroid, and from centroid to centroid?