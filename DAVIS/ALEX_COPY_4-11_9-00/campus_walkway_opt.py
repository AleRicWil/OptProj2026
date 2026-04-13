# =============================================================================
# campus_walkway_optimizer.py
# MAIN INTEGRATION SCRIPT - runs the full optimization process
# Heavy teaching comments explain every line and how it connects to
# Engineering Design Optimization Ch. 8 (Simulated Annealing).
# Euclidean distances are used everywhere (Path.length()).
# Animated SA evolution is saved as sa_evolution.gif
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import random
from loc_gen import campus, material, visitables, line_crosses_building, color_map, plot_map, doors
from nod_mac import Network, Point  # your enhanced node_machine

np.random.seed(42)  # reproducibility
# =============================================================================
# CONTROL FLAG — change this to experiment with different starting points
# =============================================================================
USE_GREEDY_INITIAL = False   # Set to False to start SA from a random valid network
USE_FULL_INITIAL = True
DEBUG_INITIAL_LAYOUT = True   # Set to True to see invalid (building-crossing)
                              # candidate connections drawn as red dashed lines.
GRID_RES = int(180)
# =============================================================================

# -----------------------------------------------------------------------------
# STEP 1: Load campus map and identify terminals (doors + POIs)
# -----------------------------------------------------------------------------
print("Loading campus map...")
campus_map, buildings_list = campus(resolution=GRID_RES, p1=(40.245751,-111.649794), p2=(40.248344,-111.646590))

# Extract all terminals that must be connected
terminal_coords = []
for spot in visitables(campus_map, "door"):
    terminal_coords.append((spot[0], spot[1]))
for spot in visitables(campus_map, "poi"):
    terminal_coords.append((spot[0], spot[1]))

print(f"Found {len(terminal_coords)} terminals (doors + POIs)")

door_to_building: dict[tuple[int, int], str] = {}

# Grab the exact coordinate bounds and scale that campus() used
y1, x1 = (40.245751, -111.649794)
y2, x2 = (40.248344, -111.646590)
miny = min(y1, y2)
minx = min(x1, x2)
w = max(y1, y2) - miny
h = max(x1, x2) - minx
mind = min(w, h)
resolution = GRID_RES                                # must match the call above
scale = int(resolution / mind)

for building_name, door_list in doors.items():
    for d in door_list:
        # convert lat/lon → grid coordinates exactly as campus() does
        y_grid = int(scale * (d[0] - miny))
        x_grid = int(scale * (d[1] - minx))
        if 0 <= y_grid < campus_map.shape[0] and 0 <= x_grid < campus_map.shape[1]:
            door_to_building[(y_grid, x_grid)] = building_name

print(f"Built same-building lookup for {len(door_to_building)} doors "
      f"(will now allow intra-building paved connections).")

print("Same-building door groups found:")
for name in set(door_to_building.values()):
    count = list(door_to_building.values()).count(name)
    print(f"  {name}: {count} doors\n")
print(door_to_building)

plot_map(campus_map)

def allowed_to_connect(p1: Point, p2: Point, campus_map, door_to_building: dict[tuple[int, int], str]) -> bool:
    """Returns True if a paved path between p1 and p2 is allowed.
    
    TEACHING CONCEPT (Engineering Design Optimization Ch. 8.4 Greedy / Ch. 8.6 SA):
    - The original line_crosses_building() treated ALL interior cells as
      forbidden → no paths could ever go through a building.
    - We now add a PHYSICALLY REALISTIC EXCEPTION: if both points are doors
      of the SAME building (according to the doors dict in loc_gen.py),
      we ALLOW the connection even if the straight line crosses interior cells.
      (Pedestrians can walk inside the building without needing exterior paving.)
    - For doors of DIFFERENT buildings (or POI-to-anything), we keep the
      strict geometric check — no building interiors may be crossed.
    """
    # Fast same-building check using the lookup we just built
    key1 = (p1.y, p1.x)
    key2 = (p2.y, p2.x)
   
    if key1 in door_to_building.keys():
        print('hoho')

    if (key1 in door_to_building and key2 in door_to_building and
        door_to_building[key1] == door_to_building[key2]):
        # Same building → always allowed (interior walk is fine)
        print('Checked doors are on same building')
        return True

    # Different buildings or POI → must pass the original strict check
    return not line_crosses_building(p1.y, p1.x, p2.y, p2.x, campus_map)

# -----------------------------------------------------------------------------
# STEP 2: Build the Network with blocked building walls
# -----------------------------------------------------------------------------
net = Network()

# Add buildings as blocked rectangular walls (prevents paved paths from crossing)
for (y1, x1), (y2, x2) in buildings_list:
    # create the four corner points with blocked material
    c1 = net.add_point(min(y1, y2), min(x1, x2), material["blocked"])
    c2 = net.add_point(min(y1, y2), max(x1, x2), material["blocked"])
    c3 = net.add_point(max(y1, y2), max(x1, x2), material["blocked"])
    c4 = net.add_point(max(y1, y2), min(x1, x2), material["blocked"])
    # connect the walls with blocked paths
    net.add_path(c1, c2, material["blocked"])
    net.add_path(c2, c3, material["blocked"])
    net.add_path(c3, c4, material["blocked"])
    net.add_path(c4, c1, material["blocked"])

# Add terminal points (doors & POIs) - these must be connected
terminals: list[Point] = []
for y, x in terminal_coords:
    terminal_point = net.determine_point(y, x)
    # force material to "door" or "poi" for nice coloring
    if net.get_point(y, x).mat not in (material["door"], material["poi"]):
        net.get_point(y, x).mat = material["door"]
    terminals.append(terminal_point)

print(f"Network initialized with {len(net.points)} total points and {len(net.paths)} paths and {len(terminals)} terminal points (buildings blocked)")

# ----------------------------------------------------------------------------- 
# STEP 3: Build INITIAL solution (greedy OR fully connected OR random — controlled by flag)
# -----------------------------------------------------------------------------
def build_greedy_initial(net: Network, terminals: list[Point], campus_map, door_to_building: dict, debug: bool = DEBUG_INITIAL_LAYOUT):
    """Greedy nearest-neighbor spanning tree (Ch. 8.4).
    
    NEW DEBUG FEATURE (for your request):
      - When debug=True, we collect EVERY pair (src, tgt) that was
        considered but REJECTED because line_crosses_building() returned True.
      - These invalid attempts are returned as a list of (Point, Point) tuples.
      - Later we plot them as red dashed lines on the campus map so you can
        visually see exactly which connections the greedy algorithm WANTED
        to make but was forced to skip because of buildings.
    
    This is perfect for debugging why certain door/POI pairs stay disconnected
    in the initial layout and for understanding the impact of the
    line_crosses_building geometric check.
    """
    # === Resolve terminals safely (same robust code you already had) ===
    term_map = {(p.y, p.x): p for p in net.points
                if p.mat in (material["door"], material["poi"])}
    resolved_terminals = [term_map[(t.y, t.x)] for t in terminals
                          if (t.y, t.x) in term_map]

    if not resolved_terminals:
        return net, []   # return empty invalid list when debugging

    connected = {resolved_terminals[0]}
    remaining = set(resolved_terminals[1:])

    invalid_attempts: list[tuple[Point, Point]] = []

    while remaining:
        best_dist = float('inf')
        best_pair = None

        for src in connected:
            for tgt in remaining:
                # === CHANGED LINE: use the new allowed check ===
                if allowed_to_connect(src, tgt, campus_map, door_to_building):
                    d = src.distance_to(tgt)
                    if d < best_dist:
                        best_dist = d
                        best_pair = (src, tgt)
                elif debug:
                    invalid_attempts.append((src, tgt))   # still record true crossings

        # ... (rest of the function unchanged)
        if best_pair is None:
            print("WARNING: Could not connect all terminals without crossing buildings!")
            break
        src, tgt = best_pair
        net.add_path(src, tgt, material["paved"])
        connected.add(tgt)
        remaining.remove(tgt)

    print(f"Greedy initial built with {len(net.paths)} paved paths "
          f"({len(connected)}/{len(resolved_terminals)} terminals connected).")

    if debug:
        print(f"DEBUG: Collected {len(invalid_attempts)} invalid (building-crossing) "
              f"candidate connections for visualization.")

    return net, invalid_attempts if debug else []

def build_fully_connected(net: Network, terminals: list[Point], campus_map, door_to_building: dict, debug: bool=DEBUG_INITIAL_LAYOUT):
    '''connects every terminal directly to every other, regardless of blocked/building interference'''
    # === Resolve terminals to THIS network's live Point objects ===
    resolved_terminals = net.resolve_terminals(terminals)
    
    if len(resolved_terminals) < 2:
        print("WARNING: Fewer than 2 terminals — nothing to connect.")
        return net, [] if debug else None

    invalid_attempts: list[tuple[Point, Point]] = []   # for debug plot

    # Use combinations-style loop (i < j) to avoid self-loops and duplicate edges
    n_term = len(resolved_terminals)
    for i in range(n_term):
        for j in range(i + 1, n_term):
            p1 = resolved_terminals[i]
            p2 = resolved_terminals[j]

            # === CHANGED: use allowed_to_connect ===
            if not allowed_to_connect(p1, p2, campus_map, door_to_building):
                if debug:
                    invalid_attempts.append((p1, p2))
                continue   # different buildings and crosses → skip

            # same-building or non-crossing → add it
            net.add_path(p1, p2, material["paved"])

    total_possible = n_term * (n_term - 1) // 2
    print(f"Fully-connected initial built with {len(net.paths)} paved paths "
          f"({total_possible} direct terminal-to-terminal connections).")

    if debug:
        print(f"DEBUG: Collected {len(invalid_attempts)} invalid (building-crossing) "
              f"candidate connections for visualization.")

    return net, invalid_attempts if debug else []

def build_random_initial(net: Network, terminals: list[Point], campus_map, door_to_building: dict, max_attempts: int = 10000) -> Network:
    """Builds a RANDOM but VALID initial network for Simulated Annealing.
    
    TEACHING PURPOSE:
    - Greedy (Ch. 8.4) always picks the shortest valid edge → very structured start.
    - Random version picks edges stochastically → explores a wider variety of
      starting topologies. This lets you study how sensitive SA is to the
      initial design (a core concept in metaheuristics).
    
    How it works:
    1. Resolve terminals to the current Network's Point objects (safety after copy()).
    2. Start with one terminal as the "connected" component.
    3. Repeatedly pick a random source (already connected) and random target
       (still unconnected). Add the paved path ONLY if it does NOT cross buildings.
    4. Stop when all terminals are connected (or we run out of attempts).
    
    Result = a random spanning-tree-like structure that is always:
        - connected
        - feasible (no building crossings)
        - uses only paved edges between existing points (Steiner points are
          still added later by SA via split_path / move_point).
    
    You can later add extra random edges or Steiner points if you want an
    even more "noisy" starting point — this version keeps it simple and
    directly comparable to the greedy tree.
    """
    # === Resolve terminals to THIS network's live Point objects ===
    # (critical after net.copy() — prevents the exact KeyError you fixed earlier)
    term_map = {(p.y, p.x): p for p in net.points
                if p.mat in (material["door"], material["poi"])}
    resolved_terminals = [term_map[(t.y, t.x)] for t in terminals
                          if (t.y, t.x) in term_map]

    if len(resolved_terminals) < 2:
        print("WARNING: Fewer than 2 terminals — nothing to connect.")
        return net

    # Start with the first terminal in the connected set
    connected = {resolved_terminals[0]}
    remaining = set(resolved_terminals[1:])

    attempts = 0
    while remaining and attempts < max_attempts:
        attempts += 1
        src = random.choice(list(connected))
        tgt = random.choice(list(remaining))

        already_exists = any(...)   # keep your existing check

        if not already_exists:
            # === CHANGED: use allowed_to_connect ===
            if allowed_to_connect(src, tgt, campus_map, door_to_building):
                net.add_path(src, tgt, material["paved"])
                connected.add(tgt)
                remaining.remove(tgt)

    if remaining:
        print(f"WARNING: Random initial only connected {len(connected)}/"
              f"{len(resolved_terminals)} terminals after {attempts} attempts.")
        print("   (Some terminals may be isolated by buildings — SA can still fix this.)")

    print(f"Random initial built with {len(net.paths)} paved paths "
          f"({len(connected)} terminals connected).")
    return net

print("\n=== BUILDING INITIAL NETWORK ===")

initial_net = net.copy()   # fresh copy so we don't mutate the building-blocked base

if USE_GREEDY_INITIAL:
    print("Using GREEDY nearest-neighbor initial solution (Ch. 8.4)...")
    initial_net, invalid_attempts = build_greedy_initial(initial_net, terminals, campus_map, door_to_building, debug=DEBUG_INITIAL_LAYOUT)
elif USE_FULL_INITIAL:
    print('Using FULLY CONNECTED valid initial solution...')
    initial_net, invalid_attempts = build_fully_connected(initial_net, terminals, campus_map, door_to_building, debug=DEBUG_INITIAL_LAYOUT)
else:
    print("Using RANDOM valid initial solution...")
    initial_net = build_random_initial(initial_net, terminals, campus_map, door_to_building)
    invalid_attempts = []

# === CRITICAL: resolve terminals to the new network's live objects ===
# (works for both all builders)
terminals = initial_net.resolve_terminals(terminals)

# Compute metrics for reporting
initial_paved = initial_net.total_paved_length()
initial_travel = initial_net.total_travel_time(terminals)
print(f"Initial paved length : {initial_paved:.1f}")
print(f"Initial total travel time: {initial_travel:.1f}")

# Budget = 1.5 × initial length (still based on whatever start we chose)
budget = initial_paved * 15
print(f"Budget set to {budget:.1f} units")
initial_net.plot_network()

# === DEBUG PLOT ===
if DEBUG_INITIAL_LAYOUT:
    print("\n=== DEBUG PLOT: Greedy layout + all rejected building-crossing candidates ===")
    initial_net.plot_initial_with_debug(campus_map, terminals, invalid_attempts,
                                        title="DEBUG: Greedy Initial + Invalid Candidates (red dashed)")

# -----------------------------------------------------------------------------
# STEP 4: Simulated Annealing (adapted from your TSP file + book Ch. 8.6)
# -----------------------------------------------------------------------------
def simulated_annealing_network(initial_net: Network, terminals: list[Point], max_iter=50000,
                                initial_temp=50000.0, cooling_rate=0.99995, budget: float = None):
    """Adapted SA for network design. State = entire Network object.
    Neighbor moves: add/remove path, move non-terminal point, split path.
    Objective = travel_time + penalty*(paved - budget)
    Uses Metropolis acceptance of WORSE moves - the core of SA (book §8.6)."""

    current_net = initial_net.copy()
    best_net = current_net.copy()
    current_obj = current_net.total_travel_time(terminals) + 1000 * max(0, current_net.total_paved_length() - budget)
    best_obj = current_obj

    temperature = initial_temp
    history = [best_obj]
    worse_accepted = 0

    # Animation setup (collect frames for GIF)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    ax.set_title("TSP-style Network SA Evolution (Live)")
    # background campus map (flipped for correct orientation)
    ax.imshow(np.flipud(campus_map), cmap=plt.cm.gray, alpha=0.3, origin='lower')
    # plot lines and points will be updated each frame
    line_objs = []  # will hold line artists for paved paths
    point_scat = ax.scatter([], [], c=[], s=30, zorder=5)
    ax.set_aspect('equal')
    plt.ion()

    frames = []  # store frame data for GIF

    print("\nStarting Simulated Annealing for walkway network...")
    print(f"Initial temperature: {temperature:.0f} | Initial objective: {current_obj:.1f}")

    for iter in range(max_iter):
        if iter % 100 == 0:
            print(f'{iter}\n')
        # --- Generate neighbor (discrete moves from node_machine) ---
        neighbor = current_net.copy()
        if not neighbor.validate_graph():
            continue

        move_type = random.choice(["add_path", "remove_path", "move_point", "split_path"])

        if move_type == "add_path" and len(neighbor.points) > 1:
            p1 = random.choice(list(neighbor.points))
            p2 = random.choice(list(neighbor.points))
            if p1 is not p2 and not any((p.p1 is p1 and p.p2 is p2) or (p.p1 is p2 and p.p2 is p1) for p in neighbor.paths):
                if allowed_to_connect(p1, p2, campus_map, door_to_building):
                    neighbor.add_path(p1, p2, material["paved"])

        elif move_type == "remove_path" and neighbor.paths:
            path_to_remove = random.choice(list(neighbor.paths))
            if path_to_remove.mat == material["paved"]:
                neighbor.remove_path(path_to_remove)

        elif move_type == "move_point":
            # move only non-terminal points (doors and POIs must stay fixed)
            movable = [p for p in neighbor.points
                       if p.mat not in (material["door"], material["poi"])]
            if movable:
                pt = random.choice(movable)
                # small random move (discrete grid)
                dy = random.randint(-2, 2)
                dx = random.randint(-2, 2)
                new_y = pt.y + dy
                new_x = pt.x + dx
                # only move if still inside map and doesn't create crossing issues
                if 0 < new_y < campus_map.shape[0] and 0 < new_x < campus_map.shape[1]:
                    if not line_crosses_building(pt.y, pt.x, new_y, new_x, campus_map):
                        neighbor.move_point(pt, new_y, new_x)

        elif move_type == "split_path" and neighbor.paths:
            paved_paths = [p for p in neighbor.paths if p.mat == material["paved"]]
            if paved_paths:
                path = random.choice(paved_paths)
                neighbor.split_path(path)

        # If invalid, skip this neighbor (cheap rejection sampling).
        if not neighbor.is_valid_space(campus_map):
            continue

        # --- Evaluate new objective ---
        new_travel = neighbor.total_travel_time(terminals)
        new_paved = neighbor.total_paved_length()
        new_obj = new_travel + 1000 * max(0, new_paved - budget)

        # --- Metropolis acceptance (exact from book and your TSP code) ---
        delta = new_obj - current_obj
        accepted = False
        if delta < 0:  # better move
            accepted = True
        else:  # worse move - accept probabilistically
            prob = np.exp(-delta / temperature)
            if random.random() < prob:
                accepted = True
                worse_accepted += 1

        if accepted:
            current_net = neighbor
            current_obj = new_obj

        # update best
        if current_obj < best_obj:
            best_net = current_net.copy()
            best_obj = current_obj

        history.append(best_obj)

        # --- Live plot & frame collection (every 500 iterations) ---
        if iter % 500 == 0 or iter == max_iter - 1:
            ax.clear()
            ax.imshow(np.flipud(campus_map), cmap=plt.cm.gray, alpha=0.3, origin='lower')
            ax.set_title(f'SA Evolution - Iter {iter} | Best obj {best_obj:.1f} | T={temperature:.1f}')

            # plot all paved paths
            for path in best_net.paths:
                if path.mat == material["paved"]:
                    ax.plot([path.p1.x, path.p2.x], [path.p1.y, path.p2.y],
                            'g-', linewidth=2, alpha=0.8)

            # plot points
            xs = [p.x for p in best_net.points]
            ys = [p.y for p in best_net.points]
            colors = [color_map.get(p.mat, 'blue') for p in best_net.points]
            ax.scatter(xs, ys, c=colors, s=40, zorder=5)

            # highlight terminals
            tx = [t.x for t in terminals]
            ty = [t.y for t in terminals]
            ax.scatter(tx, ty, c='red', s=80, marker='*', zorder=6, label='Terminals')

            ax.legend()
            plt.draw()
            plt.pause(0.001)

            # === SAFER FRAME CAPTURE FOR GIF (replaces the old buggy reshape) ===
            # This is the robust way taught in engineering visualization scripts:
            #   1. Force a full redraw
            #   2. Use buffer_rgba() — it always returns the exact current canvas size
            #   3. Reshape directly from the buffer (no guessing width/height)
            #   4. Drop the alpha channel so we keep a clean RGB array for PillowWriter
            # fig.canvas.draw()                    # make sure everything is rendered
            # rgba_buffer = fig.canvas.buffer_rgba()   # returns uint8 array of shape (h, w, 4)
            # rgba = np.frombuffer(rgba_buffer, dtype='uint8')
            # w, h = fig.canvas.get_width_height()     # get the TRUE current size
            # image = rgba.reshape((h, w, 4))[:, :, :3]   # drop alpha → RGB only
            # frames.append(image)
            # ====================================================================

        # cool temperature
        temperature *= cooling_rate
        if temperature < 1e-5:
            break

    if len(history) < 2:
        # Add the best result as a final frame if the list is too short
        history.append(best_net.copy())

    plt.ioff()
    print(f"\nSA finished. Final objective: {best_obj:.1f}")
    print(f"Worse moves accepted: {worse_accepted} (this is the SA magic!)")

    # Save animated GIF
    print("Saving SA evolution animation as sa_evolution.gif ...")
    ani = FuncAnimation(fig, lambda i: None, frames=len(frames), interval=50, repeat=False)
    writer = PillowWriter(fps=15)
    ani.save("sa_evolution.gif", writer=writer, dpi=100)
    print("GIF saved!")

    return best_net, best_obj, history


# -----------------------------------------------------------------------------
# RUN EVERYTHING
# -----------------------------------------------------------------------------
print('Running simulated annealing...')
final_net, final_obj, convergence = simulated_annealing_network(
    initial_net, terminals, max_iter=2000, budget=budget
)

initial_paved = np.pi

final_paved = final_net.total_paved_length()
final_travel = final_net.total_travel_time(terminals)

print("\n=== FINAL RESULTS (Engineering Optimization HW-style report) ===")
print(f"Initial paved length : {initial_paved:.1f} | travel time: {initial_travel:.1f}")
print(f"SA     paved length : {final_paved:.1f} | travel time: {final_travel:.1f}")
print(f"Improvement in travel time: {((initial_travel - final_travel)/initial_travel)*100:.1f}%")
print(f"Paving used vs budget: {final_paved:.1f} / {budget:.1f}")

# Final static plots
final_net.plot_network(show_labels=False)  # uses your original plotter

plt.figure()
plt.plot(convergence, 'purple', lw=2)
plt.title('Convergence - Best Objective vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Objective (travel time + penalty)')
plt.grid(True)
plt.show()

print("\nAll done! Check sa_evolution.gif for the full animated optimization process.")