"""Occupancy-map based motion prediction for an autonomous car or robot.

Pipeline (per frame):
  1. 2.5D BEV map -> agent-specific obstacle grid           (bev_occupancy.py)
  2. Euclidean distance-to-obstacle field
  3. Global route: Dijkstra on a 0.2 m grid with a clearance-aware cost
     -> auto goal (farthest reachable observed free space ahead)
     -> cost-to-go field toward that goal
  4. Local motion: sample ~1000 dynamically feasible control sequences, roll out
     the kinematic model, reject collisions / over-limit lateral accel, score the
     rest (progress via cost-to-go, clearance, unknown space, comfort)
  5. Return the top-K diverse trajectories with probabilities
     (softmax over cost) = multimodal predicted motion of the agent.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from bev_occupancy import FREE, OBSTACLE, UNKNOWN, build_occupancy


@dataclass
class PlannerWeights:
    progress: float = 1.0      # per metre of remaining cost-to-go
    clearance: float = 4.0     # exp(-(d - r)/sigma) averaged over the trajectory
    clearance_sigma: float = 0.5
    unknown: float = 3.0       # fraction of the trajectory inside unobserved cells
    lat_accel: float = 0.5     # (a_lat / a_lat_max)^2
    accel: float = 0.3         # (a / a_max)^2
    control_change: float = 2.0
    temperature: float = 1.5   # fallback softmax temperature if the agent defines none


def _sample(field, grid, x, z, oob):
    row, col = grid.world_to_cell(x, z)
    H, W = field.shape
    inside = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    out = np.full(np.shape(x), oob, dtype=float)
    out[inside] = field[row[inside], col[inside]]
    return out, inside


class MotionPredictor:
    def __init__(self, agent, weights=None, coarse=2):
        self.agent = agent
        self.w = weights or PlannerWeights()
        self.coarse = coarse

    # ------------------------------------------------------------------ maps
    def build_maps(self, map25d):
        a = self.agent
        grid = build_occupancy(map25d, min_obstacle_height=a.min_obstacle_height,
                               agent_height=a.height)
        dist = ndimage.distance_transform_edt(grid.state != OBSTACLE) * grid.res
        return grid, dist

    # ---------------------------------------------------------- global route
    def _path_clearance(self):
        a = self.agent
        if hasattr(a, "width"):
            return a.width / 2 + a.safety_margin
        return a.radius + a.safety_margin

    def global_route(self, grid, dist, start_xz, goal_xz=None):
        k = self.coarse
        H, W = grid.shape
        Hc, Wc = H // k, W // k
        res_c = grid.res * k
        dmin = dist[:Hc * k, :Wc * k].reshape(Hc, k, Wc, k).min(axis=(1, 3))
        unk = (grid.state[:Hc * k, :Wc * k] == UNKNOWN).reshape(Hc, k, Wc, k).mean(axis=(1, 3))

        r = self._path_clearance()
        blocked = dmin < r
        cell_cost = (1.0 + self.w.clearance * np.exp(-(dmin - r) / 1.0)
                     + self.w.unknown * (unk > 0.5))

        def to_node(xz):
            col = int(np.clip((xz[0] - grid.x_min) // res_c, 0, Wc - 1))
            row = int(np.clip((xz[1] - grid.z_min) // res_c, 0, Hc - 1))
            return row, col

        s_rc = to_node(start_xz)
        blocked[s_rc] = False  # the agent is standing there

        # 8-connected graph
        idx = np.arange(Hc * Wc).reshape(Hc, Wc)
        rows, cols, wts = [], [], []
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:   # dr >= 0; each pair added both ways
            r0, r1 = slice(0, Hc - dr), slice(dr, Hc)
            if dc >= 0:
                c0, c1 = slice(0, Wc - dc), slice(dc, Wc)
            else:
                c0, c1 = slice(-dc, Wc), slice(0, Wc + dc)
            ok = ~blocked[r0, c0] & ~blocked[r1, c1]
            step = res_c * np.hypot(dr, dc)
            w = step * 0.5 * (cell_cost[r0, c0] + cell_cost[r1, c1])
            a, b = idx[r0, c0][ok], idx[r1, c1][ok]
            rows += [a, b]
            cols += [b, a]
            wts += [w[ok], w[ok]]
        G = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(Hc * Wc, Hc * Wc)).tocsr()

        s_node = idx[s_rc]
        cost_from_start = dijkstra(G, indices=s_node).reshape(Hc, Wc)
        reachable = np.isfinite(cost_from_start)

        rr, cc = np.mgrid[0:Hc, 0:Wc]
        xc = grid.x_min + (cc + 0.5) * res_c
        zc = grid.z_min + (rr + 0.5) * res_c

        if goal_xz is None:
            observed_free = unk < 0.5
            within = np.hypot(xc - start_xz[0], zc - start_xz[1]) <= self.agent.goal_lookahead
            score = zc - 0.5 * np.abs(xc - start_xz[0])
            for cand in (reachable & observed_free & within, reachable & within, reachable):
                if cand.any():
                    g_rc = np.unravel_index(np.argmax(np.where(cand, score, -np.inf)), (Hc, Wc))
                    break
            goal_xz = (float(xc[g_rc]), float(zc[g_rc]))
        else:
            g_rc = to_node(goal_xz)

        g_node = idx[g_rc]
        c2g, pred = dijkstra(G, indices=g_node, return_predecessors=True)
        c2g = c2g.reshape(Hc, Wc)

        path = []
        if np.isfinite(c2g[s_rc]):
            n = s_node
            while n >= 0 and n != g_node:
                path.append(n)
                n = pred[n]
            path.append(g_node)
        path = np.array([(xc.flat[n], zc.flat[n]) for n in path]) if path else np.zeros((0, 2))

        # unreachable cells: large but ordered cost so rollouts still prefer closer ones
        finite_max = np.nanmax(np.where(np.isfinite(c2g), c2g, np.nan)) if np.isfinite(c2g).any() else 0.0
        euclid = np.hypot(xc - goal_xz[0], zc - goal_xz[1])
        c2g_filled = np.where(np.isfinite(c2g), c2g, finite_max + 3.0 * euclid + 20.0)

        return dict(goal=goal_xz, path=path, c2g=c2g_filled, reachable_goal=bool(np.isfinite(c2g[s_rc])),
                    res_c=res_c, coarse_shape=(Hc, Wc))

    def _sample_c2g(self, route, grid, x, z):
        Hc, Wc = route["coarse_shape"]
        res_c = route["res_c"]
        col = np.clip(np.floor((x - grid.x_min) / res_c).astype(int), 0, Wc - 1)
        row = np.clip(np.floor((z - grid.z_min) / res_c).astype(int), 0, Hc - 1)
        return route["c2g"][row, col]

    # -------------------------------------------------------- local rollouts
    def predict(self, map25d, v0=None, goal_xz=None, n_modes=3, maps=None):
        a, w = self.agent, self.w
        v0 = a.max_speed * 0.5 if v0 is None else float(v0)
        grid, dist = maps if maps is not None else self.build_maps(map25d)

        start = np.array([0.0, a.start_z, 0.0, v0])
        route = self.global_route(grid, dist, (0.0, max(a.start_z, 0.0) + 0.05), goal_xz)

        u = a.sample_controls(v0)
        S = a.rollout(start, u)                     # (N, T+1, 4)
        offsets, radius = a.footprint()
        x, z, psi = S[..., 0], S[..., 1], S[..., 2]

        # clearance of every footprint circle at every step
        clear = np.full(x.shape, np.inf)
        for o in offsets:
            cx, cz = x + o * np.sin(psi), z + o * np.cos(psi)
            d, _ = _sample(dist, grid, cx, cz, oob=np.inf)  # outside the map: no known obstacle
            clear = np.minimum(clear, d)
        collide = (clear < radius).any(axis=1)

        lat = a.lateral_accel(S, u)
        too_aggressive = (lat > 1.5 * a.max_lat_accel).any(axis=1)

        state_at, inside = _sample(grid.state.astype(float), grid, x, z, oob=UNKNOWN)
        unknown_frac = (state_at == UNKNOWN).mean(axis=1)

        c2g_end = self._sample_c2g(route, grid, x[:, -1], z[:, -1])
        finite_clear = np.where(np.isfinite(clear), clear, 10.0)
        clear_cost = np.exp(-(finite_clear - radius) / w.clearance_sigma).mean(axis=1)
        lat_cost = ((lat / a.max_lat_accel) ** 2).mean(axis=1)
        acc_cost = ((u[:, :, 0] / a.max_accel) ** 2).mean(axis=1)
        du = np.abs(np.diff(u[:, :, 1], axis=1)).sum(axis=1)

        J = (w.progress * c2g_end + w.clearance * clear_cost + w.unknown * unknown_frac
             + w.lat_accel * lat_cost + w.accel * acc_cost + w.control_change * du)

        feasible = ~collide & ~too_aggressive
        emergency = False
        if not feasible.any():
            # no safe motion found: brake as hard as possible, straight ahead
            emergency = True
            ub = np.zeros((1, u.shape[1], 2))
            ub[:, :, 0] = -a.max_decel
            S_e = a.rollout(start, ub)
            modes = [dict(states=S_e[0], prob=1.0, cost=float("inf"), terms={})]
        else:
            order = np.flatnonzero(feasible)[np.argsort(J[feasible])]
            min_sep = 2.0 if a.name == "car" else 0.5
            chosen = []
            for i in order:
                end = S[i, -1, :2]
                if all(np.hypot(*(end - S[j, -1, :2])) >= min_sep for j in chosen):
                    chosen.append(i)
                if len(chosen) == n_modes:
                    break
            Jc = J[chosen]
            tau = getattr(a, "mode_temperature", w.temperature)
            p = np.exp(-(Jc - Jc.min()) / tau)
            p /= p.sum()
            modes = [dict(states=S[i], controls=u[i], prob=float(pi), cost=float(J[i]),
                          terms=dict(cost_to_go=float(c2g_end[i]), clearance=float(clear_cost[i]),
                                     min_clearance_m=float(np.min(finite_clear[i]) - radius),
                                     unknown=float(unknown_frac[i]), lat_accel=float(lat_cost[i])))
                     for i, pi in zip(chosen, p)]

        return dict(
            agent=a.name, grid=grid, dist=dist, route=route, modes=modes,
            emergency_stop=emergency,
            candidates=S[feasible], n_candidates=int(len(S)), n_feasible=int(feasible.sum()),
            footprint=(offsets, radius), v0=v0, dt=a.dt,
        )
