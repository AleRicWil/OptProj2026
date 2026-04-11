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
from loc_gen import campus, material, visitables, line_crosses_building, color_map
from nod_mac import Network, Point  # your enhanced node_machine

np.random.seed(42)  # reproducibility for homework

# -----------------------------------------------------------------------------
# STEP 1: Load campus map and identify terminals (doors + POIs)
# -----------------------------------------------------------------------------
print("Loading campus map...")
campus_map, buildings_list = campus(resolution=120, p1=(40.245751,-111.649794), p2=(40.248344,-111.646590))

# Extract all terminals that must be connected
terminal_coords = []
for spot in visitables(campus_map, "door"):
    terminal_coords.append((spot[0], spot[1]))
for spot in visitables(campus_map, "poi"):
    terminal_coords.append((spot[0], spot[1]))

print(f"Found {len(terminal_coords)} terminals (doors + POIs)")

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
    term_pt = net.determine_point(y, x)
    # force material to "door" or "poi" for nice coloring
    if net.get_point(y, x).mat not in (material["door"], material["poi"]):
        net.get_point(y, x).mat = material["door"]
    terminals.append(term_pt)

print(f"Network initialized with {len(net.points)} points and {len(net.paths)} paths (buildings blocked)")

# -----------------------------------------------------------------------------
# STEP 3: Greedy initial solution (Ch. 8.4 Greedy Algorithms)
# -----------------------------------------------------------------------------
def build_greedy_initial(net: Network, terminals: list[Point], campus_map):
    """Greedy nearest-neighbor style tree: connect closest unconnected terminal
    with a straight paved path ONLY if it does not cross buildings.
    NEW: resolve terminals to THIS net's points (after copy() the Point objects
    are new). This prevents KeyError and keeps the Network invariant."""
    
    # === NEW LINES (resolve terminals by coordinate) ===
    term_map = {(p.y, p.x): p for p in net.points 
                if p.mat in (material["door"], material["poi"])}
    resolved_terminals = [term_map[(t.y, t.x)] for t in terminals 
                          if (t.y, t.x) in term_map]
    
    if not resolved_terminals:
        return net
    
    # Start with the first terminal
    connected = {resolved_terminals[0]}
    remaining = set(resolved_terminals[1:])

    while remaining:
        best_dist = float('inf')
        best_pair = None
        for src in connected:
            for tgt in remaining:
                if not line_crosses_building(src.y, src.x, tgt.y, tgt.x, campus_map):
                    d = src.distance_to(tgt)
                    if d < best_dist:
                        best_dist = d
                        best_pair = (src, tgt)
        if best_pair is None:
            print("WARNING: Could not connect all terminals without crossing buildings!")
            break
        src, tgt = best_pair
        net.add_path(src, tgt, material["paved"])
        connected.add(tgt)
        remaining.remove(tgt)

    return net

print("\n=== BUILDING GREEDY INITIAL NETWORK ===")
greedy_net = build_greedy_initial(net.copy(), terminals, campus_map)  # copy so we don't mutate original
# === FIX: make sure terminals point to the correct objects in the copied network ===
terminals = greedy_net.resolve_terminals(terminals)

greedy_paved = greedy_net.total_paved_length()
greedy_travel = greedy_net.total_travel_time(terminals)
print(f"Greedy paved length: {greedy_paved:.1f}")
print(f"Greedy total travel time: {greedy_travel:.1f}")

# Budget = 1.5 × greedy length (you can change this)
budget = greedy_paved * 1.5
print(f"Budget set to {budget:.1f} units")
greedy_net.plot_network()

# -----------------------------------------------------------------------------
# STEP 4: Simulated Annealing (adapted from your TSP file + book Ch. 8.6)
# -----------------------------------------------------------------------------
def simulated_annealing_network(initial_net: Network, terminals: list[Point], max_iter=50000,
                                initial_temp=5000.0, cooling_rate=0.99995, budget: float = None):
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
                # only add if it doesn't cross buildings
                if not line_crosses_building(p1.y, p1.x, p2.y, p2.x, campus_map):
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
    greedy_net, terminals, max_iter=80000, budget=budget
)

final_paved = final_net.total_paved_length()
final_travel = final_net.total_travel_time(terminals)

print("\n=== FINAL RESULTS (Engineering Optimization HW-style report) ===")
print(f"Greedy paved length : {greedy_paved:.1f} | travel time: {greedy_travel:.1f}")
print(f"SA    paved length : {final_paved:.1f} | travel time: {final_travel:.1f}")
print(f"Improvement in travel time: {((greedy_travel - final_travel)/greedy_travel)*100:.1f}%")
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