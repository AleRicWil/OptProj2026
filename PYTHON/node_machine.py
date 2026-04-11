from __future__ import annotations  # allows forward references
import math
from typing import Set, Tuple
import matplotlib.pyplot as plt

class Point:
    '''stores location of point, and paths connected to point'''
    def __init__(self, y: int, x: int):
        self.y = int(round(y))
        self.x = int(round(x))
        self._paths: Set[Path] = set()  # paths connected to this point

    # update location when given one
    def move(self, new_y: int, new_x: int) -> None:
        '''relocates a point to new coordinates'''
        self.y = new_y
        self.x = new_x

    def connect_path(self, path: Path) -> None:
        '''connects a path to this point'''
        self._paths.add(path)

    def incident_angles(self) -> list[tuple[Path, float]]:
        '''returns the angles of all paths connected to this point, from [0, 2*pi)'''
        return [(path, path.angle_at(self)) for path in self._paths]
    
    def ordered_paths_around(self) -> list[tuple[Path, float]]:
        '''returns all angles between all paths connected to this point, from [0, pi)'''
        return sorted(
            self.incident_angles(),
            key=lambda x: x[1]
        )
    
    def angular_gaps(self) -> list[float]:
        '''returns the angular distances between ^neighboring^ paths around this point'''
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
        '''disconnects a path from this point'''
        self._paths.discard(path)

    def distance_to(self, other: Point) -> float:
        '''measures the straight-line distance from this point to another'''
        dy = self.y - other.y
        dx = self.x - other.x
        return math.hypot(dy, dx)

    def __repr__(self) -> str:
        return f"Point({self.y}, {self.x})"


class Path:
    '''path between two points. ''' 
    def __init__(self, p1: Point, p2: Point):
        self.p1: Point = p1
        self.p2: Point = p2

        # Register with points
        p1.connect_path(self)
        p2.connect_path(self)

    def endpoints(self) -> Tuple[Point, Point]:
        '''gives tuple of endpoints of path'''
        return (self.p1, self.p2)

    def length(self) -> float:
        '''gives length of path as float'''
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

    def disconnect(self) -> None:
        '''disconnects a path from its points (the path still remembers, but the points do not)'''
        self.p1.disconnect_path(self)
        self.p2.disconnect_path(self)

    def __repr__(self) -> str:
        return f"Path({self.p1} <-> {self.p2})"


# network class contains points and paths, and allows points to be added and removed
class Network:
    def __init__(self):
        self.points: Set[Point] = set()
        self.paths: Set[Path] = set()

    def add_point(self, y: int, x: int) -> Point:
        '''add a new point to this network'''
        p = Point(y, x)
        self.points.add(p)
        return p
    
    def add_points(self, points: list[Point]):
        '''add a bunch of points to the network at once'''
        for point in points:
            self.add_point(point.y, point.x)

    def remove_point(self, point: Point) -> None:
        '''remove a point from this network'''
        # remove all connected paths first
        for path in list(point._paths):
            self.remove_path(path)

        self.points.discard(point)

    def add_path(self, p1: Point, p2: Point) -> Path:
        '''add a path between two points in this network'''
        # check p1's paths for duplicates
        for path in p1._paths:
            if p2 in (path.p1, path.p2):
                return path

        path = Path(p1, p2)
        self.paths.add(path)
        return path

    def remove_path(self, path: Path) -> None:
        '''removes one path from the network'''
        path.disconnect()
        self.paths.discard(path)

    def move_point(self, point: Point, new_y: int, new_x: int) -> None:
        '''moves a point in the network to a new location'''
        point.move(new_y, new_x)

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
    
    def get_path(net: Network, p1: Point, p2: Point) -> Path | None:
        '''given two points, return the path that corresponds to them'''
        for path in net.paths:
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

    def __repr__(self) -> str:
        return f"Network(points={len(self.points)}, paths={len(self.paths)})"
    
    def plot_network(self, show_labels=False) -> None:
        _, ax = plt.subplots()

        # draw paths
        for path in self.paths:
            x1, y1 = path.p1.x, path.p1.y
            x2, y2 = path.p2.x, path.p2.y

            ax.plot([x1, x2], [y1, y2], 'b-')

        # drawpoints
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]

        if show_labels:
            for p in self.points:
                ax.text(p.x + 0.1, p.y + 0.1, f"({p.y},{p.x})", fontsize=8)

        ax.scatter(xs, ys, c='red', s=50, zorder=3)
        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Network Graph")
        ax.grid(True)
        plt.show()


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
