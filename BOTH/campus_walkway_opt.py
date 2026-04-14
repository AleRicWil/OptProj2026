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
import random
import itertools
import collections
from loc_gen import campus, material, line_crosses_building, doors
from nod_mac import Network, Point  # your enhanced node_machine

np.random.seed(42)  # reproducibility
# =============================================================================
# CONTROL FLAGs
# =============================================================================
USE_GREEDY_INITIAL = False   # Set to False to start SA from a random valid network
USE_DENSE_INITIAL = True
VIEW_INITIAL_LAYOUT = False   # Set to True to see invalid (building-crossing)
                              # candidate connections drawn as red dashed lines.
GRID_RES = int(180)
# =============================================================================

# =============================================================================
# STEP 1: Load the campus raster map (same resolution used in the optimizer)
# =============================================================================
campus_map, buildings_list = campus(
    resolution=GRID_RES,
    p1=(40.245751, -111.649794),
    p2=(40.248344, -111.646590)
)

# =============================================================================
# STEP 2: Build the Network object
# =============================================================================
base_net = Network()

base_net.build_map(campus_map, buildings_list)
print(f"Network initialized with {len(base_net.points)} constructing points "
      f"({len(buildings_list)} buildings + {len(base_net.door_coords)} doors + {len(base_net.dest_coords)} destinations)")

base_net.connect_interiors(GRID_RES, campus_map)
print(f"Added {base_net.interior_count} interior paths "
      f"(each destination now linked to ALL doors of its building)")
print(f"There are {len(base_net.unique_pairs)} unique destination pairs.")

# ----------------------------------------------------------------------------- 
# STEP 3: Build INITIAL solution (greedy OR fully connected OR random — controlled by flag)
# -----------------------------------------------------------------------------
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

initial_net = base_net.copy()   # fresh copy so we don't mutate the building-blocked base
initial_net = build_dense_initial(initial_net, base_net.terminals, base_net.door_points, campus_map)
# Resolve terminals to the live Point objects in this network (safety after any copies)
initial_net.resolve_terminals()

print(f"Dense initial network built with "
      f"{len([p for p in initial_net.paths if p.mat == material['paved']])} paved paths.")


# Compute metrics for reporting
initial_paved = initial_net.total_paved_length()
initial_travel = initial_net.total_travel_time()
print(f"Initial paved length : {initial_paved:.1f}")
print(f"Initial total travel time: {initial_travel:.1f}")
# initial_net.plot_network(campus_map)

# ----------------------------------------------------------------------------- 
# STEP 4: NEW Simulated Annealing (dense-initial pruning + Steiner-node hubs)
# ----------------------------------------------------------------------------- 
def simulated_annealing_network(initial_net: Network, terminals: list[Point], campus_map,
                                max_iter=25000, initial_temp=1800.0, cooling_rate=0.99993,
                                target_travel_factor=1.10, weight = 25, kmax=3):
    """NEW Simulated Annealing tailored for the dense-initial walkway network.
    
    TEACHING GOAL (Engineering Design Optimization, Ch. 8.6 "Simulated Annealing"):
      Your campus walkway project is a classic *discrete network design* problem.
      Decision variables = which paved edges exist + where intermediate 'node' hubs live.
      
      We start from the DENSE initial network you built (every feasible door-door,
      dest-dest, and door-dest connection). This gives the absolute minimum possible
      pedestrian travel time, but uses a lot of paving material.
      
      SA's job is now to *minimize paved material used* while keeping travel time
      reasonable (we softly allow a controlled increase via target_travel_factor).
      This mirrors real campus-planning practice: begin with an over-connected design
      and prune intelligently.
      
      Key concepts from the book:
      • Metropolis acceptance (Eq. 8.14) lets SA accept *worse* moves when temperature
        is high — this is what lets the algorithm escape local minima and discover
        clever Steiner-node placements.
      • Temperature cools gradually (geometric schedule) so the search shifts from
        broad exploration of network topologies to fine-tuning the best designs.
      • Constraint handling is explicit and cheap: line_crosses_building() + 
        is_valid_space() reject any infeasible neighbor instantly.
      
      Why this new approach is better for your project:
      1. Terminals (doors + destinations) and interior paths are completely frozen.
      2. Only paved paths and intermediate 'node' points (material["node"]) are changed.
      3. Nodes act as natural hubs — pedestrians can converge on a node before
         crossing a large open area, often saving total paving length.
      4. Objective = paved_length + penalty_factor * max(0, travel_increase)
         This directly implements the project goal: "minimize overall pedestrian
         travel time AND for a given amount of paving material used."
      
      Moves are carefully chosen to keep the search efficient and realistic.
    """

    # Make a fresh working copy (nod_mac.py's robust deep copy guarantees
    # terminals, doors, and interior paths stay exactly as they were)
    current_net = initial_net.copy()
    best_net = current_net.copy()

    # Pre-compute all open-campus cells once (fast random node placement)
    open_locations = np.argwhere(campus_map == material["open"])
    open_locations = [tuple(loc) for loc in open_locations]

    # Baseline metrics from the dense starting network
    initial_travel = current_net.total_travel_time()
    target_travel = initial_travel * target_travel_factor   # we allow a modest degradation
    current_paved = current_net.total_paved_length()
    current_travel = initial_travel

    current_obj = 1 + weight
    best_obj = current_obj

    temperature = initial_temp
    history = [best_obj]
    worse_accepted = 0

    print("\n=== Starting Simulated Annealing (dense → pruned network) ===")
    print(f"Initial paved length : {current_paved:.1f} units")
    print(f"Initial total travel time : {current_travel:.1f} units")
    print(f"Target travel time (allowed) : {target_travel:.1f} ({target_travel_factor-1:.0%} increase)")
    print(f"Initial temperature : {temperature:.0f} | Objective : {current_obj:.1f}\n")

    for iter in range(max_iter):
        # Generate a neighbor by copying the current network
        neighbor = current_net.copy()

        # Choose one discrete move (the five moves that let SA explore the design space)
        move_types = ["add_node", "remove_node", "move_node", "break_intersection", "add_path", "remove_path"]
        move_weights = [0.2, 0.1, 0.2, 0.1, 0.1, 0.3] 

        # Inside your SA loop:
        move_type = random.choices(move_types, weights=move_weights, k=1)[0]

        if move_type == "add_node" and open_locations:
            # Add a brand-new Steiner node (hub) at a random open campus location.
            # This is how we introduce extra connection points that can reduce total paving.
            y, x = random.choice(open_locations)
            new_node = neighbor.add_point(y, x, material["node"])

            # find local candidates
            candidates = [p for p in neighbor.points if (p is not new_node and (p.mat != material["blocked"]))]
            if not candidates:
                continue
            
            # sort based on distance, and pick local-ish points
            candidates.sort(key=lambda p: (p.y - y)**2 + (p.x - x)**2)
            k = random.randint(2, kmax)
            pool = candidates[:min(len(candidates), k * 3)]
            selected = random.sample(pool, min(len(pool), k))

            # make connections as found
            connections_made = 0
            for p in selected:
                if not line_crosses_building(new_node.y, new_node.x, p.y, p.x, campus_map):
                    neighbor.add_path(new_node, p, material["paved"])
                    connections_made += 1

            # don't even bother if the node isn't fully integrated
            if connections_made < 2:
                neighbor.remove_point(new_node)

        elif move_type == "break_intersection":
            grid_size = 40
            buckets = collections.defaultdict(list)
            
            for path in neighbor.paths:
                # Find the bounding box of the path
                min_x = min(path.p1.x, path.p2.x)
                max_x = max(path.p1.x, path.p2.x)
                min_y = min(path.p1.y, path.p2.y)
                max_y = max(path.p1.y, path.p2.y)
                
                # Identify which grid cells this path touches
                start_col, end_col = int(min_x // grid_size), int(max_x // grid_size)
                start_row, end_row = int(min_y // grid_size), int(max_y // grid_size)
                
                for r in range(start_row, end_row + 1):
                    for c in range(start_col, end_col + 1):
                        buckets[(r, c)].append(path)

            # 2. Filter for buckets that actually have potential intersections
            candidate_buckets = [paths for paths in buckets.values() if len(paths) >= 2]
            
            if not candidate_buckets:
                continue
            
            # 3. Pick a random candidate area and check for a real intersection
            random.shuffle(candidate_buckets)
            found_and_split = False
            
            for path_list in candidate_buckets[:5]: # Only check a few areas to keep SA fast
                for p1, p2 in itertools.combinations(path_list, 2):
                    # Call your existing intersection logic
                    new_node = neighbor.split_on_intersection(p1, p2)
                    if new_node:
                        found_and_split = True
                        break
                if found_and_split:
                    break
            
            if not found_and_split:
                continue


        elif move_type == "move_node":
            # Move an existing node (only intermediate nodes can move — terminals stay fixed).
            # Small random jitter helps the optimizer "slide" hubs into better positions.
            node_pts = [p for p in neighbor.points if p.mat == material["node"]]
            if node_pts:
                pt = random.choice(node_pts)
                dy = random.randint(-10, 10)
                dx = random.randint(-10, 10)
                new_y, new_x = pt.y + dy, pt.x + dx
                if (0 <= new_y < campus_map.shape[0] and 
                    0 <= new_x < campus_map.shape[1] and
                    campus_map[new_y, new_x] not in (material["blocked"], material["interior"])):
                    neighbor.move_point(pt, new_y, new_x)

        elif move_type == "remove_node":
            # Remove an intermediate node (and all its paved paths).
            # This is the opposite of add_node and lets SA aggressively prune useless hubs.
            node_pts = [p for p in neighbor.points if p.mat == material["node"]]
            if node_pts:
                pt = random.choice(node_pts)
                neighbor.remove_point(pt)   # nod_mac.py automatically cleans up incident paths

        elif move_type == "add_path":
            # Occasionally add a new paved connection (useful early when temperature is high).
            # We only add if it does NOT cross any building.
            potentials = (material["door"], material["poi"], material["node"], material["destination"])
            pts = [p for p in neighbor.points if p.mat in potentials]
            for _ in range(12):   # limited random trials for efficiency
                p1 = random.choice(pts)
                p2 = random.choice(pts)
                if p1 is not p2 and neighbor.get_path(p1, p2) is None:
                    if not line_crosses_building(p1.y, p1.x, p2.y, p2.x, campus_map):
                        neighbor.add_path(p1, p2, material["paved"])
                        break

        elif move_type == "remove_path":
            # The main pruning move — remove a paved path to save material.
            # This is what directly reduces the objective.
            paved_paths = [p for p in neighbor.paths if p.mat == material["paved"]]
            if paved_paths:
                path = random.choice(paved_paths)
                neighbor.remove_path(path)

        # After any move, reject the neighbor immediately if it violates space constraints
        if not neighbor.is_valid_space(campus_map):
            # may want to redo this to backtrack one step rather than progress and diminish temperature
            continue

        # reject immediately if no longer connected
        if not neighbor.is_connected([p for p in neighbor.points if p.mat == material["destination"]]):
            continue


        # Evaluate the new candidate solution
        new_paved = neighbor.total_paved_length()
        new_travel = neighbor.total_travel_time()

        if new_paved > initial_paved:
            continue

        paved_ratio = new_paved / initial_paved
        travel_ratio = new_travel / initial_travel

        new_obj = paved_ratio + weight*travel_ratio

        # Metropolis acceptance criterion (the heart of SA — see book §8.6)
        delta = new_obj - current_obj
        accepted = False
        if delta < 0:                     # always accept improvement
            accepted = True
        else:                             # accept worse move probabilistically
            prob = np.exp(-delta / temperature)
            if random.random() < prob:
                accepted = True
                worse_accepted += 1

        if accepted:
            # remove any disconnected nodes for this one
            # node_pts = [p for p in current_net.points if p.mat == material["node"]]
            # for pt in node_pts:
            #     # count incident paths
            #     degree = len(pt._paths)
            #     if degree < 2:
            #         best_net.remove_point(pt)


            current_net = neighbor
            current_obj = new_obj
            current_paved = new_paved
            current_travel = new_travel

            

        # Update best solution found so far
        if current_obj < best_obj:
            best_net = current_net.copy()
            best_obj = current_obj
            neighbor.plot_network(campus_map, save_flag=True, idx=iter)

        history.append(best_obj)

        # Cool the temperature (geometric schedule)
        temperature *= cooling_rate
        if temperature < 0.01:
            break

        if iter % 500 == 0 and iter > 0:
            print(f"Iter {iter:5d} | Temp {temperature:6.2f} | "
                  f"Best {best_obj:.2f} | Curr {current_obj:.2f} | Paved {current_paved:6.1f} | "
                  f"Travel {current_travel:6.1f}")

    print(f"\nSA finished after {iter} iterations.")
    print(f"Final objective: {best_obj:.1f}")
    print(f"Worse moves accepted: {worse_accepted}  ← this is the SA magic that finds good hubs!")

    return best_net, best_obj, history


# ----------------------------------------------------------------------------- 
# RUN EVERYTHING (updated for the new pruning-focused SA)
# ----------------------------------------------------------------------------- 
print('\nRunning simulated annealing on the dense initial network (pruning mode)...')

final_net, final_obj, convergence = simulated_annealing_network(
    initial_net, 
    initial_net.terminals,          # destinations only (doors are fixed inside the network)
    campus_map,
    max_iter=8000,
    initial_temp=2000.0,
    cooling_rate=0.99996,
    target_travel_factor=1.20,       # allow up to 20% travel-time increase
    weight = 0.1,                    # how much to emphasize travel time over pavement
    kmax=3,                          # max number of nearby points a node could possibly connect to
)

final_paved = final_net.total_paved_length()
final_travel = final_net.total_travel_time()

print("\n=== FINAL RESULTS (Engineering Optimization HW-style report) ===")
print(f"Initial (dense) paved length : {initial_paved:.1f} | travel time: {initial_travel:.1f}")
print(f"SA-optimized paved length   : {final_paved:.1f} | travel time: {final_travel:.1f}")
print(f"Paving material saved       : {initial_paved - final_paved:.1f} units "
      f"({((initial_paved - final_paved)/initial_paved)*100:.1f}%)")
if initial_travel > 0:
    print(f"Travel-time change         : {((final_travel - initial_travel)/initial_travel)*100:.1f}%")

# Final static plots (uses your existing plotter)
final_net.plot_network(campus_map)

plt.figure(figsize=(10, 6))
plt.plot(convergence, 'darkgreen', linewidth=2.5)
plt.title('Simulated Annealing Convergence\n'
          'Objective = Paved Length + Travel-Time Penalty')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.grid(True, alpha=0.6)
plt.show()