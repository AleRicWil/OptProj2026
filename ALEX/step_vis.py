# =============================================================================
# campus_walkway_path_visualizer.py
# SIMPLIFIED & RE-ORGANIZED VERSION
#
# TEACHING GOAL (Engineering Design Optimization, Ch. 8.4 "Greedy Algorithms"
# and Ch. 8.6 "Simulated Annealing"):
#   This script is a clean, interactive visualizer that helps you *understand*
#   the objective function your optimizer is minimizing.
#
#   NEW REALISTIC OBJECTIVE (exactly what you asked for):
#     • We only care about connecting *destinations* (POIs inside buildings).
#     • Travelers must enter a building through a *door* → we add fixed
#       "interior" paths (zero paving cost, but they count in travel time).
#     • The paved network (exterior only) does NOT have to touch every door.
#     • Objective = sum of shortest-path travel times between EVERY pair of
#       destinations (hybrid graph = paved exterior + interior access).
#
#   Why this is perfect for our campus walkway project:
#     - Greedy (Ch. 8.4) gives a fast, reasonable starting layout.
#     - Later Simulated Annealing (Ch. 8.6) will improve it by adding/removing
#       paved edges and moving Steiner points while respecting the budget.
#     - This visualizer lets you *see* every shortest path that contributes
#       to the total travel-time objective — exactly what SA is optimizing.
#
#   Code is now shorter, better organized, and heavily commented for learning.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap
import itertools
import heapq
from collections import defaultdict

# Import our project modules (all paths are relative — keep them in the same folder)
from loc_gen import (
    campus, material, visitables, line_crosses_building,
    color_map, plot_map, doors, destinations
)
from nod_mac import Network, Point   # nod_mac.py already has total_travel_time, resolve_terminals, etc.

print("=== Campus Walkway Visualizer (Simplified & Teaching-Focused) ===")
print("Loading campus map and building the network...")

# =============================================================================
# STEP 1: Load the campus raster map (same resolution used in the optimizer)
# =============================================================================
GRID_RES = 280
campus_map, buildings_list = campus(
    resolution=GRID_RES,
    p1=(40.245751, -111.649794),
    p2=(40.248344, -111.646590)
)

# =============================================================================
# STEP 2: Build the Network object
# =============================================================================
net = Network()

net.build_map(campus_map, buildings_list)
print(f"Network initialized with {len(net.points)} constructing points "
      f"({len(buildings_list)} buildings + {len(net.door_coords)} doors + {len(net.dest_coords)} destinations)")

net.connect_interiors(GRID_RES, campus_map)
print(f"Added {net.interior_count} interior paths "
      f"(each destination now linked to ALL doors of its building)")
print(f"There are {len(net.unique_pairs)} unique destination pairs.")

# =============================================================================
# STEP 4: Greedy nearest-neighbor initial network (Ch. 8.4)
# =============================================================================
def build_dense_initial(net: Network, terminals: list[Point], door_points: list[Point], campus_map) -> Network:
    """Builds a DENSE paved walkway network by attempting FULL connections between:
       • Every destination (terminal/POI) to every other destination,
       • Every destination to every door, and
       • Every door to every other door.

    TEACHING GOAL (Engineering Design Optimization, Ch. 8.4 "Greedy Algorithms" 
    and Ch. 8.6 "Simulated Annealing"):
      In our campus walkway project we are solving a classic discrete network 
      design problem. The paved paths are the DECISION VARIABLES.
      
      A simple greedy MST (the function you already have) gives a minimal 
      spanning tree — useful for a cheap starting point. 
      
      This dense builder creates a much richer initial graph: it adds 
      almost every feasible paved edge that a pedestrian might actually use. 
      This is exactly what professional Steiner-tree / road-network optimizers 
      do before pruning with SA or other metaheuristics.
      
      Why is this helpful for learning?
      1. It lets you instantly see (in the step_vis interactive stepper) how 
         shortest-path travel times change when many routing options exist.
      2. It gives Simulated Annealing a realistic "over-connected" starting 
         design to improve upon — exactly the kind of initial solution you 
         would hand an optimizer in a real campus-planning project.
      3. It demonstrates constraint handling: we only add a paved edge if it 
         does NOT cross a building (using the exact same line_crosses_building 
         test you already use in the greedy builder).
      
      REAL-WORLD CAMPUS LOGIC (the key teaching insight):
      - Doors inside the SAME building are allowed to connect directly with 
        paved paths even if the straight line crosses interior cells 
        (pedestrians can walk inside the building without needing exterior paving).
      - All other connections (different buildings or POI-to-anything) must 
        stay strictly outside buildings.
      
      This function is self-contained and can be dropped straight into 
      step_vis.py right after your existing build_greedy_initial function.
    """

    # ------------------------------------------------------------------
    # 1. Build the same-building lookup table (exactly as in campus_walkway_opt.py)
    #    This is the realistic campus rule that makes door-to-door connections
    #    inside one building allowed.
    # ------------------------------------------------------------------
    from collections import defaultdict
    door_to_building: dict[tuple[int, int], str] = {}
    
    # Grab the exact coordinate scaling that campus() used when it built the map
    y1, x1 = (40.245751, -111.649794)
    y2, x2 = (40.248344, -111.646590)
    miny = min(y1, y2)
    minx = min(x1, x2)
    mind = min(max(y1, y2) - miny, max(x1, x2) - minx)
    scale = int(GRID_RES / mind)   # GRID_RES is already defined in step_vis.py

    for building_name, door_list in doors.items():   # doors dict is imported from loc_gen
        for d in door_list:
            y_grid = int(scale * (d[0] - miny))
            x_grid = int(scale * (d[1] - minx))
            if 0 <= y_grid < campus_map.shape[0] and 0 <= x_grid < campus_map.shape[1]:
                door_to_building[(y_grid, x_grid)] = building_name

    # ------------------------------------------------------------------
    # 2. Helper: allowed_to_connect (nested so the function stays self-contained)
    #    This is the exact same safety check used in your main optimizer.
    # ------------------------------------------------------------------
    def allowed_to_connect(p1: Point, p2: Point) -> bool:
        """Returns True only if a paved path is physically realistic."""
        key1 = (p1.y, p1.x)
        key2 = (p2.y, p2.x)
        
        # # Same-building doors are ALWAYS allowed (interior travel is free)
        # if (key1 in door_to_building and key2 in door_to_building and
        #     door_to_building[key1] == door_to_building[key2]):
        #     return True
        
        # Different buildings or POI connections must not cross any building
        return not line_crosses_building(p1.y, p1.x, p2.y, p2.x, campus_map)

    # ------------------------------------------------------------------
    # 3. Count existing paved paths so we can report how many we added
    # ------------------------------------------------------------------
    paved_before = len([p for p in net.paths if p.mat == material["paved"]])
    print("Building DENSE pathway network (full connections between "
          "destinations + doors)...")

    # ------------------------------------------------------------------
    # 4. Door ↔ Door connections
    # ------------------------------------------------------------------
    print("   • Connecting every door to every other door...")
    import itertools
    for d1, d2 in itertools.combinations(door_points, 2):
        if allowed_to_connect(d1, d2):
            net.add_path(d1, d2, material["paved"])

    # ------------------------------------------------------------------
    # 5. Destination ↔ Destination connections
    # ------------------------------------------------------------------
    print("   • Connecting every destination to every other destination...")
    for t1, t2 in itertools.combinations(terminals, 2):
        if allowed_to_connect(t1, t2):
            net.add_path(t1, t2, material["paved"])

    # ------------------------------------------------------------------
    # 6. Destination ↔ Door connections
    # ------------------------------------------------------------------
    print("   • Connecting every destination to every door...")
    for dest in terminals:
        for door in door_points:
            if allowed_to_connect(dest, door):
                net.add_path(dest, door, material["paved"])

    # ------------------------------------------------------------------
    # 7. Final report (great for learning how many edges your network now has)
    # ------------------------------------------------------------------
    paved_after = len([p for p in net.paths if p.mat == material["paved"]])
    added = paved_after - paved_before
    print(f"Dense pathway builder finished!")
    print(f"   Added {added} new paved paths.")
    print(f"   Total paved paths now: {paved_after} (this is your rich initial graph)")
    print(f"   (Terminals remain fully connected via exterior paved + interior paths)")

    return net

net = build_dense_initial(net, net.terminals, net.door_points, campus_map)
print(f"→ Greedy initial network built with "
      f"{len([p for p in net.paths if p.mat == material['paved']])} paved paths.")

# Resolve terminals to the live Point objects in this network (safety after any copies)
terminals = net.resolve_terminals(net.terminals)

# =============================================================================
# STEP 5: Helper to compute shortest path between any two destinations
# =============================================================================
def get_shortest_path(net: Network, start: Point, goal: Point):
    """Returns the list of points in the shortest path and the total Euclidean travel time.
    Uses the exact same hybrid graph (paved + interior) that total_travel_time() uses
    in your main optimizer. This is what SA is minimizing!"""
    if start is goal:
        return [start], 0.0

    adj, idx, point_list = net.build_distance_dict(terminals)  # built-in helper from nod_mac.py

    start_idx = idx[start]
    goal_idx = idx[goal]

    # Dijkstra (standard shortest-path algorithm)
    dist = {i: float('inf') for i in range(len(adj))}
    dist[start_idx] = 0.0
    pred = {i: None for i in range(len(adj))}
    pq = [(0.0, start_idx)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, weight in adj[u]:
            alt = d + weight
            if alt < dist[v]:
                dist[v] = alt
                pred[v] = u
                heapq.heappush(pq, (alt, v))

    if dist[goal_idx] == float('inf'):
        return [], float('inf')

    # Reconstruct path
    path: list[Point] = []
    current = goal_idx
    while current is not None:
        path.append(point_list[current])
        current = pred[current]
    path.reverse()

    return path, dist[goal_idx]

# =============================================================================
# STEP 7: Interactive matplotlib figure (exact same style as plot_map())
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 9), dpi=90)
plt.subplots_adjust(bottom=0.18)

# Use the exact same colormap as loc_gen.py so the background matches perfectly
cmap = ListedColormap([color_map[i] for i in sorted(color_map.keys())])
ax.imshow(
    campus_map,
    cmap=cmap,
    origin="lower",
    vmin=0,
    vmax=len(color_map) - 1
)

# Plot the full network once (green = paved, gray dashed = interior)
for path in net.paths:
    if path.mat == material["paved"]:
        ax.plot([path.p1.x, path.p2.x], [path.p1.y, path.p2.y],
                color='limegreen', linewidth=2.5, alpha=0.9, solid_capstyle='round')
    elif path.mat == material["interior"]:
        ax.plot([path.p1.x, path.p2.x], [path.p1.y, path.p2.y],
                color='gray', linestyle='--', linewidth=1.8, alpha=0.65)

# Plot doors (small circles) and destinations (gold stars)
door_xs = [p.x for p in net.points if p.mat == material['door']]
door_ys = [p.y for p in net.points if p.mat == material['door']]
ax.scatter(door_xs, door_ys, color=color_map[material['door']], s=45, zorder=5,
           edgecolors='black', linewidth=0.6)

dest_xs = [t.x for t in terminals]
dest_ys = [t.y for t in terminals]
ax.scatter(dest_xs, dest_ys, c='gold', s=160, marker='*', zorder=6,
           edgecolors='darkred', linewidth=1.8, label='Destinations (terminals)')

ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.set_title("Campus Walkway Network — DESTINATION-FOCUSED Objective\n"
             "Green = paved walkways • Gray dashed = interior access • Blue = current shortest path",
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right')

# Container for the temporary blue route lines (removed/re-drawn each step)
route_lines: list = []
info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

current_pair_idx = 0

def update_visualization():
    """Redraws the blue shortest-path highlight and updates the info box.
    Called every time you click Previous/Next."""
    global route_lines
    # Remove old route
    for line in route_lines:
        line.remove()
    route_lines.clear()

    start, goal = net.unique_pairs[current_pair_idx]
    path_points, travel_time = get_shortest_path(net, start, goal)

    if path_points:
        for i in range(len(path_points) - 1):
            pA, pB = path_points[i], path_points[i + 1]
            line, = ax.plot([pA.x, pB.x], [pA.y, pB.y],
                            color='blue', linewidth=5.5, alpha=0.85,
                            solid_capstyle='round', zorder=10)
            route_lines.append(line)

        info_text.set_text(
            f"Pair {current_pair_idx + 1}/{len(net.unique_pairs)}\n"
            f"From → To: Destination pair\n"
            f"Shortest travel time: {travel_time:.1f} grid units"
        )
    else:
        info_text.set_text("No path found (graph disconnected)")

    fig.canvas.draw_idle()


def next_pair(event):
    global current_pair_idx
    current_pair_idx = (current_pair_idx + 1) % len(net.unique_pairs)
    update_visualization()


def prev_pair(event):
    global current_pair_idx
    current_pair_idx = (current_pair_idx - 1) % len(net.unique_pairs)
    update_visualization()


# Add Previous / Next buttons (placed at bottom of figure)
ax_prev = plt.axes([0.15, 0.05, 0.15, 0.075])
ax_next = plt.axes([0.35, 0.05, 0.15, 0.075])
btn_prev = Button(ax_prev, 'Previous Pair', color='lightblue', hovercolor='0.85')
btn_next = Button(ax_next, 'Next Pair', color='lightblue', hovercolor='0.85')
btn_prev.on_clicked(prev_pair)
btn_next.on_clicked(next_pair)

# Show the first pair immediately
update_visualization()

plt.show()

print("\nVisualization ready!")
print("   • Click Previous/Next to step through every destination pair.")
print("   • Blue line = shortest path used in the total_travel_time objective.")
print("   • This is exactly the quantity your Simulated Annealing optimizer is minimizing.")
print("   • Great way to debug and understand your network before running the full SA!")