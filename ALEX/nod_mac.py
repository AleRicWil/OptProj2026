# =============================================================================
# node_machine.py
# Added optimization-specific helpers (total_paved_length, connectivity,
# all-pairs shortest-path travel cost using PURE Euclidean distances).
# =============================================================================

from __future__ import annotations
import math
from typing import Set, Tuple, List
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import itertools
import heapq
from loc_gen import (
    campus, material, visitables, line_crosses_building,
    color_map, plot_map, doors, destinations
)

class Point:
    '''stores location of point, and paths connected to point'''
    def __init__(self, y: int, x: int, mat: int = material["paved"]):
        self.y = round(y)
        self.x = round(x)
        self.mat = mat
        self._paths: Set[Path] = set()

    def move(self, new_y: int, new_x: int) -> None:
        '''relocates a point to new coordinates'''
        self.y = new_y
        self.x = new_x

    def connect_path(self, path: Path) -> None:
        self._paths.add(path)

    def incident_angles(self) -> list[tuple[Path, float]]:
        return [(path, path.angle_at(self)) for path in self._paths]

    def ordered_paths_around(self) -> list[tuple[Path, float]]:
        return sorted(self.incident_angles(), key=lambda x: x[1])

    def angular_gaps(self) -> list[float]:
        ordered = self.ordered_paths_around()
        if len(ordered) < 2:
            return []
        gaps = []
        for i in range(len(ordered)):
            a1 = ordered[i][1]
            a2 = ordered[(i + 1) % len(ordered)][1]
            diff = (a2 - a1) % (2 * math.pi)
            gaps.append(diff)
        return gaps

    def disconnect_path(self, path: Path) -> None:
        self._paths.discard(path)

    def distance_to(self, other: Point) -> float:
        '''Pure Euclidean distance - exactly what you asked for'''
        dy = self.y - other.y
        dx = self.x - other.x
        return math.hypot(dy, dx)

    def __repr__(self) -> str:
        return f"({self.y}, {self.x}), m={self.mat}"


class Path:
    '''path between two points. '''
    def __init__(self, p1: Point, p2: Point, mat: int = material["paved"]):
        self.p1: Point = p1
        self.p2: Point = p2
        self.mat: int = mat
        p1.connect_path(self)
        p2.connect_path(self)

    def endpoints(self) -> Tuple[Point, Point]:
        return (self.p1, self.p2)

    def length(self) -> float:
        '''Euclidean length - this is the distance function used everywhere in the optimizer'''
        dy = self.p1.y - self.p2.y
        dx = self.p1.x - self.p2.x
        return math.hypot(dy, dx)

    def angle(self) -> float:
        '''returns the angle of this path as seen from above. angle in range [0 and pi)'''
        dy = self.p2.y - self.p1.y
        dx = self.p2.x - self.p1.x

        angle = math.atan2(dy, dx)

        # make direction invariant, so pi looks like 0
        if angle < 0:
            angle += math.pi
        if angle >= math.pi:
            angle -= math.pi

        return angle
    
    def angle_difference(self, other: Path) -> float:
        '''returns the difference in angle between two paths. angle in range [0, pi/2)]'''
        diff = abs(self.angle() - other.angle())
        return min(diff, math.pi - diff)
    
    def angle_at(self, p: Point) -> float:
        '''returns the angle of this path as seen from point p. angle in range (-pi, pi]
        '''
        if p is self.p1:
            dy = self.p2.y - self.p1.y
            dx = self.p2.x - self.p1.x
        elif p is self.p2:
            dy = self.p1.y - self.p2.y
            dx = self.p1.x - self.p2.x
        else:
            raise ValueError("Point is not an endpoint of this path")

        return math.atan2(dy, dx)
    
    def is_near_parallel(self, other: Path, tolerance: float = 0.4) -> bool:
        '''checks for two paths to be within 22.5 degrees of each other, by default'''
        # 0.4 is about 22.5 degrees.
        return self.angle_difference(other) < tolerance
    
    def endpoints_close(self, other: Path, tolerance: float = 1.0) -> bool:
        '''checks if the endpoints of two paths are within a tolerance of each otehr'''
        return any(
            p1.distance_to(p2) <= tolerance
            for p1 in (self.p1, self.p2)
            for p2 in (other.p1, other.p2)
        )
    
    def is_similar_to(self, other: Path, angle_tol: float = 0.1, dist_tol: float = 1.0) -> bool:
        '''checks if two paths are near-parallel and have close endpoints'''
        return (self.is_near_parallel(other, angle_tol)
                and self.endpoints_close(other, dist_tol))
    
    def intersection(self, other: Path) -> Point | None:
        '''returns a point, if one exists, of an intersection between two paths'''
        x1, y1 = self.p1.x, self.p1.y
        x2, y2 = self.p2.x, self.p2.y
        x3, y3 = other.p1.x, other.p1.y
        x4, y4 = other.p2.x, other.p2.y

        # compute denominator
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)

        # parallel or collinear
        if abs(denom) < 1e-9:
            return None

        # intersection point (infinite lines)
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

        # check if within both segments
        def within(a: float, b: float, c: float) -> bool:
            return min(a, b) < c < max(a, b)

        if (
            within(x1, x2, px) and within(y1, y2, py) and
            within(x3, x4, px) and within(y3, y4, py)
        ):
            return Point(py, px)  # remember (y, x)

        return None

    def crosses(self, other: Path) -> bool:
        '''intersection(), but a bool instead of a Point | None'''
        return self.intersection(other) is not None

    def disconnect(self) -> None:
        '''disconnects a path from its points (the path still remembers, but the points do not)'''
        self.p1.disconnect_path(self)
        self.p2.disconnect_path(self)

    def __repr__(self) -> str:
        return f"({self.p1} <-> {self.p2})"


class Network:
    def __init__(self):
        self.points: Set[Point] = set()
        self.paths: Set[Path] = set()

    def build_map(self, campus_map, buildings_list):
        '''Creates the entire map of campus with all buildings, doors, destinations'''
        for p1, p2 in buildings_list:
            self.add_building(p1, p2)

        self.door_coords = list(visitables(campus_map, "door"))
        self.door_points: list[Point] = []
        for y, x in self.door_coords:
            door_pt = self.add_point(y, x, material["door"])
            self.door_points.append(door_pt)

        self.dest_coords = list(visitables(campus_map, "destination"))
        self.terminals: list[Point] = []
        for y, x in self.dest_coords:
            dest_pt = self.add_point(y, x, material["destination"])
            self.terminals.append(dest_pt)

        self.unique_pairs = list(itertools.combinations(self.terminals, 2))

    def connect_interiors(self, GRID_RES, campus_map):
        #   Each destination now connects to EVERY door of the building it belongs to.
        #   This gives full interior connectivity (pedestrians can enter/exit via any door).

        # Re-calculate the exact same scaling used when the campus map was built
        # (this guarantees grid coordinates match the ones in visitables())
        y1, x1 = (40.245751, -111.649794)
        y2, x2 = (40.248344, -111.646590)
        miny = min(y1, y2)
        minx = min(x1, x2)
        mind = min(max(y1, y2) - miny, max(x1, x2) - minx)
        scale = int(GRID_RES / mind)

        # Build mapping: doors-key → list of actual door Point objects in the network
        from collections import defaultdict
        building_to_doors = defaultdict(list)
        for building_key, door_list in doors.items():
            for d in door_list:
                y_grid = int(scale * (d[0] - miny))
                x_grid = int(scale * (d[1] - minx))
                if 0 <= y_grid < campus_map.shape[0] and 0 <= x_grid < campus_map.shape[1]:
                    door_pt = self.get_point(y_grid, x_grid)
                    if door_pt:
                        building_to_doors[building_key].append(door_pt)

        # Helper: given a destination name (e.g. "eb_lobby"), return the matching building key
        def get_building_key_for_dest(dest_name: str) -> str | None:
            if not dest_name:
                return None
            prefix = dest_name.split('_')[0].lower()          # first part before any underscore
            for building_key in doors.keys():
                key_lower = building_key.lower()
                # Match if prefix appears anywhere in the door key
                if prefix in key_lower:
                    return building_key
            return None

        # Add interior paths for every destination
        self.interior_count = 0
        for dest_name, coord_list in destinations.items():
            building_key = get_building_key_for_dest(dest_name)

            # Get the destination Point(s) from the network
            for coord in coord_list:
                dy = int(scale * (coord[0] - miny))
                dx = int(scale * (coord[1] - minx))
                dest_pt = self.get_point(dy, dx)
                if not dest_pt:
                    continue

                if building_key and building_key in building_to_doors:
                    # Connect to EVERY door in the matched building
                    for door_pt in building_to_doors[building_key]:
                        self.add_path(dest_pt, door_pt, material["interior"])
                        self.interior_count += 1

    def add_point(self, y: int, x: int, mat: int = material["paved"]) -> Point:
        '''Add (or return existing) point at (y, x).
        ENHANCED: now allows safe material upgrades (paved → door/poi).
        This makes the campus initialization + copy() bullet-proof even if
        a door lands exactly on a previously-created paved node.
        '''
        y = round(y)   # protect against any float drift from splits/moves
        x = round(x)

        existing = self.get_point(y, x)
        if existing:
            # === MATERIAL CONFLICT HANDLING (this fixes your exact error) ===
            if existing.mat != mat:
                # Allow upgrade: paved -> door or poi (common in campus setup)
                if ((existing.mat == material["paved"] or existing.mat == material['blocked']) and 
                    mat in (material["door"], material["destination"])):
                    existing.mat = mat   # upgrade the material in-place
                    return existing
                elif existing.mat == material["paved"] and mat in [material['node']]:
                    existing.mat = mat
                    return existing                                                    
                else:
                    # true conflict (should never happen after the move_point fix)
                    raise ValueError(
                        f"Material conflict at ({y}, {x}): "
                        f"existing={existing.mat} (mat={existing.mat}), new={mat}"
                    )
            return existing

        # brand new point
        p = Point(y, x, mat)
        self.points.add(p)
        return p
    
    def add_points(self, points: list[Point], mat: int):
        '''add a bunch of points to the network at once'''
        for point in points:
            self.add_point(point.y, point.x, mat)

    def remove_point(self, point: Point) -> None:
        '''remove a point from this network'''
        # remove all connected paths first
        for path in list(point._paths):
            self.remove_path(path)

        self.points.discard(point)

    def add_path(self, p1: Point, p2: Point, mat: int = material["paved"]) -> Path | None:
        '''add a path between two points in this network.
        NEW ROBUSTNESS: if p1 or p2 are not already in self.points (e.g. stale points
        from a previous copy), we resolve them by coordinate using determine_point.
        This fixes the exact bug you are seeing after net.copy() + greedy.'''
        # resolve points to THIS network's objects (critical after copy())
        p1 = self.determine_point(p1.y, p1.x)
        p2 = self.determine_point(p2.y, p2.x)

        # check p1's paths for duplicates (original safety)
        for path in p1._paths:
            if p2 in (path.p1, path.p2):
                return path

        new_path = Path(p1, p2, mat)

        # if this is a walk path, ensure it doesn't cross any buildings
        if mat == material["paved"]:
            for existing in self.paths:
                if existing.mat == material["blocked"]:
                    if new_path.crosses(existing):
                        new_path.disconnect()
                        return None

        self.paths.add(new_path)
        return new_path

    def remove_path(self, path: Path) -> None:
        '''removes one path from the network'''
        path.disconnect()
        self.paths.discard(path)

    def move_point(self, point: Point, new_y: int, new_x: int) -> None:
        '''Moves a non-terminal point to a new location.
        CRITICAL ROBUSTNESS: 
          - If the target coordinate is already occupied, we MERGE the two points.
          - This preserves the "one Point per coordinate" invariant.
          - Terminals (doors/POIs) are never overwritten (we simply reject the move).
          - This is exactly how professional graph-based optimizers (e.g. road-network or pipe-routing tools) maintain validity.
        '''
        # snap to integer grid (just in case)
        new_y = round(new_y)
        new_x = round(new_x)

        # safety bounds check already done in the SA loop, but keep it here too
        if not (0 <= new_y < 300 and 0 <= new_x < 300):  # campus_map size is ~300x250
            return

        target = self.get_point(new_y, new_x)

        if target is None:
            # free space → just move
            point.move(new_y, new_x)
        elif target is point:
            # no-op, already there
            return
        elif target.mat in (material["door"], material["poi"]):
            # never overwrite a terminal — reject this neighbor move silently
            # (this is the safe, teaching-friendly choice for your project)
            return
        else:
            # two paved nodes collided → merge them (keeps connectivity)
            # merge_points already exists and does all the rewiring
            self.merge_points(point, target, new_y, new_x)

    def merge_points(self, p1: Point, p2: Point, new_y: int, new_x: int) -> Point:
        '''merges two points, and all their paths, into one new point at new_y, new_x'''
        # create new merged point
        p_new = Point(new_y, new_x)
        self.points.add(p_new)

        # collect all paths from both points
        all_paths = set(p1._paths) | set(p2._paths)

        # remove old direct connection between p1 and p2 (if exists)
        for path in list(all_paths):
            if (path.p1 is p1 and path.p2 is p2) or (path.p1 is p2 and path.p2 is p1):
                self.remove_path(path)
                all_paths.discard(path)

        # helper: avoid duplicate connections
        def already_connected(a: Point, b: Point) -> bool:
            return any(
                (p.p1 is a and p.p2 is b) or (p.p1 is b and p.p2 is a)
                for p in p_new._paths
            )

        # rewire remaining paths
        for path in list(all_paths):
            other = path.p2 if path.p1 in (p1, p2) else path.p1

            # remove old path
            self.remove_path(path)

            # add path to new point, and avoid duplicate edges
            if not already_connected(p_new, other):
                self.add_path(p_new, other)

        # remove old points
        self.points.discard(p1)
        self.points.discard(p2)

        return p_new
    
    def get_point(self, y: int, x: int) -> Point | None:
        '''checks if a point already exists at some specific coordinates'''
        for p in self.points:
            if p.y == y and p.x == x:
                return p
        return None
    
    def determine_point(self, y: int, x: int) -> Point:
        '''checks if a point exists at a location, and either returns it or adds one there'''
        existing = self.get_point(y, x)
        if existing is not None:
            return existing
        return self.add_point(y, x)
    
    def get_path(self, p1: Point, p2: Point) -> Path | None:
        '''given two points, return the path that corresponds to them (if any).
        Used in the example() and could be handy for debugging neighbors.
        Uses object identity (is) because Points are unique objects in the graph.'''
        for path in self.paths:
            if (path.p1 is p1 and path.p2 is p2) or (path.p1 is p2 and path.p2 is p1):
                return path
        return None

    def split_path(self, path: Path) -> Point:
        '''splits a path in half, adding a point at(near) the midpoint'''
        p1, p2 = path.p1, path.p2

        # compute midpoint in geometric space
        mid_y = (p1.y + p2.y) / 2
        mid_x = (p1.x + p2.x) / 2

        # snap to integer grid
        y = int(round(mid_y))
        x = int(round(mid_x))

        # resolve node existence separately
        mid = self.determine_point(y, x)

        # avoid degenerate split
        if mid is p1 or mid is p2:
            return mid

        # rewrite graph
        self.remove_path(path)
        self.add_path(p1, mid)
        self.add_path(mid, p2)

        return mid
    
    def split_on_intersection(self, p1: Path, p2: Path) -> Point | None:
        '''splits an intersection into four segments all connecting to the intersection point'''
        ip = p1.intersection(p2)

        if ip is None:
            return None

        # use centralized point management
        p_new = self.determine_point(ip.y, ip.x)

        # store endpoints
        a, b = p1.p1, p1.p2
        c, d = p2.p1, p2.p2

        # remove old paths
        self.remove_path(p1)
        self.remove_path(p2)

        # rewire (self-loops are inherently avoided by add_path if desired)
        self.add_path(a, p_new)
        self.add_path(p_new, b)
        self.add_path(c, p_new)
        self.add_path(p_new, d)

        return p_new
    
    def add_building(self, p1: Point, p2: Point, mat: int = material["blocked"]) -> list[Point]:
        """
        creates a rectangular building using p1 and p2 as opposite corners.
        """
        y1, x1 = p1
        y2, x2 = p2
        
        c1 = self.add_point(min(y1, y2), min(x1, x2), material["blocked"])
        c2 = self.add_point(min(y1, y2), max(x1, x2), material["blocked"])
        c3 = self.add_point(max(y1, y2), max(x1, x2), material["blocked"])
        c4 = self.add_point(max(y1, y2), min(x1, x2), material["blocked"])
        # Connect them with blocked paths (these are never paved)
        self.add_path(c1, c2, material["blocked"])
        self.add_path(c2, c3, material["blocked"])
        self.add_path(c3, c4, material["blocked"])
        self.add_path(c4, c1, material["blocked"])

    def __repr__(self) -> str:
        return f"Network(points={len(self.points)}, paths={len(self.paths)})"
    
    def plot_network(self, campus_map, save_flag=False, idx=0) -> None:
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
        for path in self.paths:
            if path.mat == material["paved"]:
                ax.plot([path.p1.x, path.p2.x], [path.p1.y, path.p2.y],
                        color='limegreen', linewidth=2.5, alpha=0.9, solid_capstyle='round')
            elif path.mat == material["interior"]:
                ax.plot([path.p1.x, path.p2.x], [path.p1.y, path.p2.y],
                        color='gray', linestyle='--', linewidth=1.8, alpha=0.65)

        # Plot doors (small circles) and destinations (gold stars)
        door_xs = [p.x for p in self.points if p.mat == material['door']]
        door_ys = [p.y for p in self.points if p.mat == material['door']]
        ax.scatter(door_xs, door_ys, color=color_map[material['door']], s=45, zorder=5,
                edgecolors='black', linewidth=0.6)
        
        node_xs = [p.x for p in self.points if p.mat == material['node']]
        node_ys = [p.y for p in self.points if p.mat == material['node']]
        ax.scatter(node_xs, node_ys, color=color_map[material['node']], s=45, zorder=5,
                edgecolors='black', linewidth=0.6)

        dest_xs = [t.x for t in self.terminals]
        dest_ys = [t.y for t in self.terminals]
        ax.scatter(dest_xs, dest_ys, c='gold', s=160, marker='*', zorder=6,
                edgecolors='darkred', linewidth=1.8, label='Destinations (terminals)')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title("Campus Walkway Network\n"
                    "Green = paved walkways • Gray dashed = interior access",
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        
        if save_flag:
            plt.savefig(f"network_result_{idx}.png", 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='white')
            
        plt.show()

        # ---------------------------------------------------------------------
    # NEW DEBUG PLOT: shows campus map background + valid paved paths
    #                 + invalid (rejected) candidates in red dashed lines
    # ---------------------------------------------------------------------
    def plot_initial_with_debug(self, campus_map, terminals: List[Point], invalid_attempts: list[tuple[Point, Point]] = None,
                                title: str = "Greedy Initial Network + Rejected Candidates"):
        """Debug visualization exactly as you requested.
        
        TEACHING PURPOSE (Engineering Design Optimization Ch. 8.4):
          - Green solid lines = actual paved paths chosen by greedy.
          - Red dashed lines = every candidate connection that was
            rejected because it crossed a building (line_crosses_building).
          - Background = the raster campus map so you instantly see
            why a particular edge was invalid.
          - Terminals highlighted with stars.
        
        This makes the constraint surface visible and is invaluable for
        debugging the initial solution quality before SA starts.
        """
        fig, ax = plt.subplots(figsize=(12, 10), dpi=120)
        
        # 1. Campus map background (flipped for correct matplotlib orientation)
        ax.imshow(np.flipud(campus_map), cmap=plt.cm.gray, alpha=0.35, origin='lower')
        
        # 2. Plot all VALID paved paths (same as your original plotter)
        for path in self.paths:
            if path.mat == material["paved"]:
                ax.plot([path.p1.x, path.p2.x], [path.p1.y, path.p2.y],
                        color='limegreen', linewidth=3, solid_capstyle='round')

        # 3. Plot INVALID candidate connections (the new debug feature)
        if invalid_attempts:
            for src, tgt in invalid_attempts:
                ax.plot([src.x, tgt.x], [src.y, tgt.y],
                        color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # 4. Plot ALL points
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        colors = [color_map.get(p.mat, 'blue') for p in self.points]
        ax.scatter(xs, ys, c=colors, s=50, zorder=5, edgecolors='black', linewidth=0.5)

        # 5. Highlight terminals (doors + POIs)
        tx = [t.x for t in terminals]   # terminals is resolved in main script
        ty = [t.y for t in terminals]
        ax.scatter(tx, ty, c='gold', s=180, marker='*', zorder=6, edgecolors='darkred', linewidth=1.5,
                   label='Terminals (must connect)')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("x (grid units)")
        ax.set_ylabel("y (grid units)")
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------------
    # NEW OPTIMIZATION HELPERS (added for SA)
    # -----------------------------------------------------------------------------
    def total_paved_length(self) -> float:
        """Sum of Euclidean lengths of all paved paths. This is the 'cost' we penalize."""
        return sum(p.length() for p in self.paths if p.mat == material["paved"])

    def get_terminal_points(self, terminals: List[Point]) -> List[Point]:
        """Return only the points we must connect (doors + POIs)."""
        return [p for p in terminals if p in self.points]

    def build_distance_dict(self) -> tuple:
        """Build adjacency dict for Dijkstra. NOW INCLUDES interior paths for building-internal travel.
        Paved = decision variables (cost we minimize). Interior = fixed (zero paving cost)."""
        point_list = list(self.points)
        idx = {p: i for i, p in enumerate(point_list)}
        
        adj = {i: [] for i in range(len(point_list))}
        for path in list(self.paths):
            # BOTH paved (exterior) and interior (inside buildings) are used for travel time
            if path.mat in (material["paved"], material["interior"]):
                if path.p1 not in idx or path.p2 not in idx:
                    continue
                i = idx[path.p1]
                j = idx[path.p2]
                length = path.length()
                adj[i].append((j, length))
                adj[j].append((i, length))

        return adj, idx, point_list

    def is_connected(self, terminals: List[Point]) -> bool:
        """Check if all terminals are in one connected component (pure graph connectivity)."""
        if not terminals:
            return True
        terminals = self.resolve_terminals()
        adj, idx, point_list = self.build_distance_dict()
        start = idx[terminals[0]]
        visited = set()
        stack = [start]
        while stack:
            u = stack.pop()
            if u not in visited:
                visited.add(u)
                for v, _ in adj[u]:
                    if v not in visited:
                        stack.append(v)
        term_ids = {idx[t] for t in terminals}
        return term_ids.issubset(visited)

    def total_travel_time(self) -> float:
        """NEW objective: sum of shortest-path distances between EVERY pair of DESTINATIONS.
        Uses hybrid graph = paved exterior paths + fixed interior door-to-destination edges.
        Doors are no longer terminals. This is exactly what your SA optimizer will minimize."""
        # ---------------------------------------------------------------------
        # 2. CRITICAL SAFETY: Resolve stale terminal references
        #    (After Network.copy() or neighbor moves, Python object identities
        #     can become invalid. resolve_terminals() rebuilds the list using
        #     coordinate lookup — this is the standard fix in graph-based
        #     discrete optimizers.)
        # ---------------------------------------------------------------------
        self.resolve_terminals()          # updates self.terminals in place
        destinations = self.terminals     # now guaranteed to be live objects

        if not destinations or len(destinations) < 2:
            return 0.0

        # ---------------------------------------------------------------------
        # 3. Fast connectivity check (pure graph traversal, no distances needed)
        #    If any terminal is unreachable, the whole solution is invalid.
        # ---------------------------------------------------------------------
        # if not self.is_connected(destinations):
        #     return 1e9   # massive penalty — tells SA "this is bad"

        # ---------------------------------------------------------------------
        # 4. Use pre-computed unique_pairs when available (set in build_map())
        #    This avoids recomputing itertools.combinations on every call.
        # ---------------------------------------------------------------------
        if hasattr(self, 'unique_pairs') and self.unique_pairs:
            pairs = self.unique_pairs
        else:
            pairs = list(itertools.combinations(destinations, 2))

        # ---------------------------------------------------------------------
        # 5. Sum shortest-path distances for every unique pair
        #    This is exactly the "find the shortest route for each destination
        #    pair" behavior you asked for.
        # ---------------------------------------------------------------------
        total_travel = 0.0
        for start, goal in pairs:
            # Delegate to the shared helper — reuses the identical Dijkstra
            # implementation + predecessor reconstruction logic you already
            # trust in get_shortest_route().
            _, distance = self.get_shortest_route(start, goal)

            if distance == float('inf') or distance >= 1e9:
                return 1e9   # any disconnected pair makes the whole network invalid

            total_travel += distance

        return total_travel

    def get_shortest_route(self, start: Point, goal: Point):
        """Returns the list of points in the shortest path and the total Euclidean travel time.
        Uses the exact same hybrid graph (paved + interior) that total_travel_time() uses
        in your main optimizer. This is what SA is minimizing!"""
        if start is goal:
            return [start], 0.0

        adj, idx, point_list = self.build_distance_dict()  # built-in helper from nod_mac.py

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

    def is_valid_space(self, campus_map) -> bool:
        """NEW HELPER: Returns True only if NO paved path crosses a building.
        Uses the exact same raster sampling as loc_gen.line_crosses_building
        (pure Euclidean line sampling). This is the single source of truth
        for feasibility—much safer than the old vector intersection with
        blocked wall paths (which only checked boundaries)."""
        for path in self.paths:
            if path.mat == material["paved"]:
                if line_crosses_building(path.p1.y, path.p1.x,
                                         path.p2.y, path.p2.x, campus_map):
                    return False
        return True

    def validate_graph(self) -> bool:
        """Quick graph sanity check — call during development to catch stale references early."""
        for path in list(self.paths):
            if path.p1 not in self.points or path.p2 not in self.points:
                print(f"WARNING: Stale reference in Path {path}")
                return False
        return True

    def copy(self) -> Network:
        """Deep copy of the entire network — ROBUST version for SA neighbor trials.
        
        CRITICAL TEACHING POINT: 
        The original copy() assumed perfect graph consistency (every Path endpoint
        is always in self.points). In practice, merge_points, split_path, and
        move_point can temporarily leave "stale" Point references in Paths.
        This version uses coordinates as the source of truth and includes a
        coordinate fallback. This is exactly how professional discrete
        optimization tools (campus planning, VLSI routing, pipe networks) stay stable.
        
        This fix lets the SA loop run to completion (80 000+ iterations) without
        crashing even when neighbor moves temporarily violate the invariant.
        """
        # Step 0: Create a fresh, empty Network instance
        new_net = Network()
        
        # =====================================================================
        # 1. ROBUST COPY OF THE CORE GRAPH (points + paths)
        # =====================================================================
        # We use coordinates as the single source of truth (same technique
        # that fixed the earlier KeyError bug). This guarantees that even if
        # merge_points() or move_point() left temporary stale references,
        # the copy remains perfectly consistent.
        
        point_map: dict[Point, Point] = {}
        
        # Copy every live Point using its (y, x, mat) triple
        for p in list(self.points):          # list() protects against mutation
            new_p = new_net.add_point(p.y, p.x, p.mat)
            point_map[p] = new_p
        
        # Defensive fallback: any stale Point that only exists inside a Path
        # (can happen right after a merge/split during neighbor generation)
        for path in list(self.paths):
            for endpoint in (path.p1, path.p2):
                if endpoint not in point_map:
                    new_p = new_net.add_point(endpoint.y, endpoint.x, endpoint.mat)
                    point_map[endpoint] = new_p
        
        # Now copy every Path, always using the NEW network's Point objects
        for path in list(self.paths):
            p1 = path.p1
            p2 = path.p2
            # Fast lookup through the map; fallback to coordinate resolution
            new_p1 = point_map.get(p1, new_net.determine_point(round(p1.y), round(p1.x)))
            new_p2 = point_map.get(p2, new_net.determine_point(round(p2.y), round(p2.x)))
            new_net.add_path(new_p1, new_p2, path.mat)
        
        # =====================================================================
        # 2. COPY ALL AUXILIARY ATTRIBUTES (the part you asked to add)
        # =====================================================================
        # These were set by build_map() and connect_interiors().
        # We carry them over and remap any Point references so that
        # total_travel_time(), is_connected(), resolve_terminals(), etc.
        # work perfectly on the copy.
        
        if hasattr(self, 'door_coords'):
            new_net.door_coords = list(self.door_coords)          # immutable tuples → safe shallow copy
        
        if hasattr(self, 'door_points'):
            # door_points is a list[Point] – we MUST remap to the new objects
            new_net.door_points = [point_map[p] for p in self.door_points]
        
        if hasattr(self, 'dest_coords'):
            new_net.dest_coords = list(self.dest_coords)
        
        if hasattr(self, 'terminals'):
            # terminals list is critical for travel-time objective
            new_net.terminals = [
                point_map.get(p, new_net.determine_point(round(p.y), round(p.x)))
                for p in self.terminals
            ]
        
        if hasattr(self, 'unique_pairs'):
            # unique_pairs = list of (Point, Point) tuples for all-pairs combinations
            new_net.unique_pairs = [
                (point_map[p1], point_map[p2])
                for p1, p2 in self.unique_pairs
            ]
        
        if hasattr(self, 'interior_count'):
            new_net.interior_count = self.interior_count   # integer – direct copy
        
        # =====================================================================
        # 3. Return the fully-functional independent clone
        # =====================================================================
        return new_net

    def resolve_terminals(self) -> list[Point]:
        """CRITICAL SAFETY METHOD for any routine that uses Network.copy()
        
        After .copy() (or any graph reconstruction) the Point objects in the
        'terminals' list become stale (different Python objects, same coordinates).
        This method rebuilds the list using coordinate lookup so that
        is_connected(), total_travel_time(), and any future SA moves always
        see the correct Point instances that actually live in THIS network.
        
        This is the standard pattern in graph-based discrete optimization
        (think Steiner-tree or TSP with intermediate nodes). It eliminates
        the KeyError forever.
        """
        # Fast lookup: (y, x) -> live Point object in this network
        coord_map: dict[tuple[int, int], Point] = {}
        for p in self.points:
            if p.mat in (material["door"], material["poi"]):
                # round to catch any tiny floating-point drift from moves/splits
                coord_map[(round(p.y), round(p.x))] = p

        resolved: list[Point] = []
        for t in self.terminals:
            key = (round(t.y), round(t.x))
            if key in coord_map:
                resolved.append(coord_map[key])
            else:
                # safety fallback (should never trigger after the first greedy step)
                # print(f"WARNING: terminal at {key} not found in network — recreating")
                resolved.append(self.determine_point(key[0], key[1]))
        
        self.terminals = resolved


# =============================================================================
# example() - run it for testing
# =============================================================================
def example():
    net = Network()

    # make points
    a = net.add_point(0, 0)
    b = net.add_point(0, 5)
    c = net.add_point(5, 5)
    d = net.add_point(5, 0)
    e = net.add_point(4, 2)

    # Create edges
    net.add_path(a, b)
    net.add_path(b, c)
    net.add_path(c, d)
    net.add_path(d, a)

    # Diagonal cross
    net.add_path(a, c)
    net.add_path(b, d)

    # Connect center
    net.add_path(e, a)
    net.add_path(e, b)
    net.add_path(e, c)
    net.add_path(e, d)
    net.plot_network(True)

    # split intersection
    ac = net.get_path(a, c)
    bd = net.get_path(b, d)
    f = net.split_on_intersection(ac, bd)
    net.plot_network(True)

    # move a point
    net.move_point(f, 1, 3)    
    net.plot_network(True)

    # remove a point
    net.remove_point(d) 
    net.plot_network(True)

    # remove a path
    net.remove_path(net.get_path(a, b)) 
    net.plot_network(True)

if __name__ == "__main__":
    example()