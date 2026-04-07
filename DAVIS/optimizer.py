import pulp
from location_generator import *
from path_generator import *

def compute_turn_angle(a, b, c):
    '''For handling the 'sharp corner' penalty'''
    v1 = (a[0]-b[0], a[1]-b[1])
    v2 = (c[0]-b[0], c[1]-b[1])

    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = m.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = m.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 == 0 or mag2 == 0:
        return 0

    cos_theta = dot / (mag1 * mag2)
    cos_theta = max(-1, min(1, cos_theta))

    angle = m.acos(cos_theta)
    return angle  # radians


def paths_too_similar(p1, p2, threshold=10):
    '''Like the name says'''
    set1 = set(p1)
    set2 = set(p2)
    overlap = len(set1 & set2)
    return overlap > threshold


def apply_solution(grid, edges, selected_edges):
    '''Takes suggestions and names. in that order'''
    for e_id in selected_edges:
        for (x, y) in edges[e_id]["path"]:
            if grid[x][y] == material["open"]:
                grid[x][y] = material["paved"]
    return grid


def optimize_network(pois, edges, lambda_edges=0.0, lambda_turn=0.0, alpha=10.0):
    '''Crazy optimizer for doing all the work. doesn't work, but gives me ideas. Check out PuLP'''
    prob = pulp.LpProblem("PathOptimization", pulp.LpMinimize)

    # --- VARIABLES ---
    x = {}  # edge selection
    for e_id, e in enumerate(edges):
        x[e_id] = pulp.LpVariable(f"x_{e_id}", cat="Binary")

    # multi-commodity flow
    f = {}
    n = len(pois)

    for s in range(n):
        for t in range(n):
            if s == t:
                continue
            for e_id, e in enumerate(edges):
                f[(s,t,e_id)] = pulp.LpVariable(f"f_{s}_{t}_{e_id}", lowBound=0, upBound=1)


    # check for and incentivize shared edges
    usage = {}

    for e_id in x:
        usage[e_id] = pulp.lpSum([
            f[(s,t,e_id)]
            for s in range(n)
            for t in range(n)
            if s != t
        ])


    # check for and penalize similar edges
    for e1_id, e1 in enumerate(edges):
        for e2_id, e2 in enumerate(edges):
            if e1_id >= e2_id:
                continue
            if paths_too_similar(e1["path"], e2["path"]):
                prob += x[e1_id] + x[e2_id] <= 1

    # --- OBJECTIVE ---
    prob += pulp.lpSum([
        e["cost"] * x[e_id]
        for e_id, e in enumerate(edges)
    ]) \
    + lambda_edges * pulp.lpSum([x[e_id] for e_id in x]) \
    - alpha * pulp.lpSum([usage[e_id] for e_id in x])

    # flow conservation
    for s in range(n):
        for t in range(n):
            if s == t:
                continue

            for v in range(n):
                inflow = []
                outflow = []

                for e_id, e in enumerate(edges):
                    if e["j"] == v:
                        inflow.append(f[(s,t,e_id)])
                    if e["i"] == v:
                        outflow.append(f[(s,t,e_id)])

                if v == s:
                    prob += pulp.lpSum(outflow) - pulp.lpSum(inflow) == 1
                elif v == t:
                    prob += pulp.lpSum(inflow) - pulp.lpSum(outflow) == 1
                else:
                    prob += pulp.lpSum(inflow) - pulp.lpSum(outflow) == 0

    for e_id in x:
        prob += pulp.lpSum([
            f[(s,t,e_id)]
            for s in range(n)
            for t in range(n)
            if s != t
        ]) <= (n-1) * x[e_id]
                
    prob += pulp.lpSum([x[e_id] for e_id in x]) >= n - 1

    turn_penalty = []

    for e1_id, e1 in enumerate(edges):
        for e2_id, e2 in enumerate(edges):
            if e1["j"] == e2["i"]:  # connected

                a = pois[e1["i"]]
                b = pois[e1["j"]]
                c = pois[e2["j"]]

                angle = compute_turn_angle(a, b, c)

                if angle > 0:
                    y = pulp.LpVariable(f"turn_{e1_id}_{e2_id}", cat="Binary")

                    prob += y <= x[e1_id]
                    prob += y <= x[e2_id]

                    # don't count the angle as a cost if less than 45 degree turn
                    angle_cost = angle
                    if angle < np.pi/4:
                        angle_cost = 0

                    turn_penalty.append(angle_cost * y)

    prob += lambda_turn * pulp.lpSum(turn_penalty)

    prob.solve(pulp.PULP_CBC_CMD(msg=1))

    selected_edges = [e_id for e_id in x if pulp.value(x[e_id]) > 0.5]

    return selected_edges


if __name__ == "__main__":
    grid = campus(128, (40.245751,-111.649794), (40.248344,-111.646590))
    # grid = simple_space(10, 20, "corners")
    plot_map(grid)

    pois, edges = build_candidate_graph(grid, k=3)

    selected = optimize_network(pois, edges,
                                lambda_edges=2.0,
                                lambda_turn=5.0)

    grid = apply_solution(grid, edges, selected)

    plot_map(grid)