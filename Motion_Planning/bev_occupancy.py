"""Turn a 2.5D pseudo-LiDAR BEV map into a metric obstacle occupancy grid.

Input format (produced by Pseudo_Lidar/src/compression/map_visual.py + voxel_to_25d.py):
    array [1, 2, H=512, W=208], float32
    row  i -> forward distance  z = Z_MIN + (i + 0.5) * RES   (Z_MIN = 0 m)
    col  j -> lateral distance  x = X_MIN + (j + 0.5) * RES   (X_MIN = -10 m, +x = right)
    ch 0 -> y_min of the points in the cell, normalized: y_norm = (y + 10) / 20
    ch 1 -> y_max of the points in the cell, same normalization
    Camera frame: y points DOWN, so y_min = highest point, y_max = lowest point.
    Cells with no points were written as y = 0 -> y_norm = 0.5 in BOTH channels.

Output: an OccupancyGrid with per-cell height above the fitted ground plane and a
state per cell: FREE (observed ground), OBSTACLE, UNKNOWN (never observed).
Obstacles depend on the agent: a cell is an obstacle only if something sticks up
above `min_obstacle_height` AND its lowest point is below the agent's height
(so tree canopies / signs above a small robot do not block it).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# Must match map_visual.py
RES = 0.1
X_RANGE = (-10.0, 10.0)
Y_RANGE = (-10.0, 10.0)
Z_RANGE = (0.0, 50.0)
EMPTY_VALUE = 0.5          # normalized value of y = 0, used for empty cells
KITTI_CAMERA_HEIGHT = 1.65  # m, cam2 above ground; fallback if ground fit fails
STEREO_FB = 721.5 * 0.54    # focal (px) x baseline (m) of KITTI cam2/cam3: depth sigma ~ z^2 / FB per px of disparity error

FREE, OBSTACLE, UNKNOWN = 0, 1, 2


@dataclass
class OccupancyGrid:
    state: np.ndarray            # [H, W] uint8 in {FREE, OBSTACLE, UNKNOWN}
    height_top: np.ndarray       # [H, W] m above ground of highest point (nan if unknown)
    height_bottom: np.ndarray    # [H, W] m above ground of lowest point (nan if unknown)
    ground_plane: tuple          # (a, b, c): ground y = a*x + b*z + c (camera frame)
    res: float = RES
    x_min: float = X_RANGE[0]
    z_min: float = Z_RANGE[0]
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.state.shape

    @property
    def extent(self):
        """matplotlib imshow extent with origin='lower': (x0, x1, z0, z1)."""
        H, W = self.shape
        return (self.x_min, self.x_min + W * self.res, self.z_min, self.z_min + H * self.res)

    def world_to_cell(self, x, z):
        col = np.floor((np.asarray(x) - self.x_min) / self.res).astype(np.int64)
        row = np.floor((np.asarray(z) - self.z_min) / self.res).astype(np.int64)
        return row, col

    def cell_to_world(self, row, col):
        x = self.x_min + (np.asarray(col) + 0.5) * self.res
        z = self.z_min + (np.asarray(row) + 0.5) * self.res
        return x, z


def load_25d(path):
    arr = np.load(path)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.shape[0] != 2:
        raise ValueError(f"{path}: expected 2 channels (y_min, y_max), got shape {arr.shape}")
    return arr.astype(np.float32)


def _denorm(y_norm):
    return y_norm * (Y_RANGE[1] - Y_RANGE[0]) + Y_RANGE[0]


def fit_ground_plane(x, z, y_low, iters=200, inlier_tol=0.15, seed=0):
    """RANSAC plane y = a*x + b*z + c on the lowest point of each observed cell.

    Only cells near the expected ground height are considered as candidates, so
    tall walls / cars do not pull the plane up.
    """
    cand = np.abs(y_low - KITTI_CAMERA_HEIGHT) < 0.6
    if cand.sum() < 50:
        return (0.0, 0.0, KITTI_CAMERA_HEIGHT), 0
    xs, zs, ys = x[cand], z[cand], y_low[cand]
    A = np.stack([xs, zs, np.ones_like(xs)], axis=1)
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        idx = rng.choice(len(xs), 3, replace=False)
        try:
            p = np.linalg.solve(A[idx], ys[idx])
        except np.linalg.LinAlgError:
            continue
        # reject steep planes (|slope| > ~11 degrees)
        if abs(p[0]) > 0.2 or abs(p[1]) > 0.2:
            continue
        n = int((np.abs(A @ p - ys) < inlier_tol).sum())
        if n > best_n:
            best, best_n = p, n
    if best is None:
        return (0.0, 0.0, KITTI_CAMERA_HEIGHT), 0
    inl = np.abs(A @ best - ys) < inlier_tol
    best, *_ = np.linalg.lstsq(A[inl], ys[inl], rcond=None)  # refine on inliers
    return tuple(float(v) for v in best), int(inl.sum())


def remove_see_through(obstacle, ground, xc, zc, h_top, cam_h, min_obstacle_height,
                       agent_height, bin_deg=0.5, min_ground_hits=3):
    """Visibility check that undoes a weakness of the 2.5D (y_min, y_max) encoding.

    A cell holding road points plus a few stereo outliers 3 m up looks like a solid
    0-3 m column. But if the camera sees road *behind* that cell, the sight line to
    that road passed through the cell at some height h; nothing can be there.
    For a sight line from camera height Hc to ground at range r2, its height when
    crossing range r1 is  h = Hc * (1 - r1 / r2).  If road is observed at ranges r2
    whose sight lines cross the cell inside the band [min_obstacle_height, h_hi],
    with h_hi = min(cell top, agent height), the band is empty -> not an obstacle.
    """
    bearing = np.degrees(np.arctan2(xc, zc))
    rng = np.hypot(xc, zc)
    bins = np.floor(bearing / bin_deg).astype(np.int64)

    g_bins, g_rng = bins[ground], rng[ground]
    order = np.lexsort((g_rng, g_bins))
    g_bins, g_rng = g_bins[order], g_rng[order]
    starts = np.searchsorted(g_bins, np.unique(g_bins))
    lookup = {b: g_rng[s:e] for b, s, e in zip(np.unique(g_bins), starts,
                                                list(starts[1:]) + [len(g_bins)])}

    oi = np.flatnonzero(obstacle)
    o_bins, o_rng = bins.flat[oi], rng.flat[oi]
    h_lo = min_obstacle_height
    h_hi = np.minimum(np.minimum(h_top.flat[oi], agent_height), 0.98 * cam_h)
    # Road "seen" just behind an object is usually the object's own depth smear, not a gap:
    # only count road farther than 2 sigma of stereo depth noise (sigma ~ r^2 / (f * B)).
    smear = np.maximum(1.0, 2.0 * o_rng ** 2 / STEREO_FB)
    r2_lo = np.maximum(o_rng / (1.0 - h_lo / cam_h), o_rng + smear)
    r2_hi = o_rng / (1.0 - h_hi / cam_h)

    clear = np.zeros(len(oi), bool)
    for k in range(len(oi)):
        g = lookup.get(o_bins[k])
        if g is None or h_hi[k] <= h_lo:
            continue
        n = np.searchsorted(g, r2_hi[k], side="right") - np.searchsorted(g, r2_lo[k], side="left")
        clear[k] = n >= min_ground_hits
    out = obstacle.copy()
    out.flat[oi[clear]] = False
    return out, int(clear.sum())


def build_occupancy(map25d, min_obstacle_height=0.3, agent_height=2.0,
                    min_blob_cells=4, blob_range_coef=0.02, blob_link_cells=1, blob_link_min_range=20.0, max_obstacle_height=4.0,
                    see_through_check=True):
    """Classify every BEV cell for one agent.

    min_obstacle_height : bumps lower than this are drivable (curbs ~0.15 m)
    agent_height        : overhangs whose lowest point is above this are passable
    min_blob_cells      : obstacle blobs smaller than this are treated as stereo noise
    blob_range_coef     : extra cells per m^2 of range added to that threshold
    blob_link_cells     : far fragments within 2*this many cells count as one blob
    blob_link_min_range : only fragments beyond this range (m) are linked; near specks stay separate
    max_obstacle_height : points higher than this (sky / depth outliers) are ignored
    """
    y_min_n, y_max_n = map25d[0], map25d[1]
    H, W = y_min_n.shape
    observed = ~((y_min_n == EMPTY_VALUE) & (y_max_n == EMPTY_VALUE))

    rows, cols = np.mgrid[0:H, 0:W]
    xc = X_RANGE[0] + (cols + 0.5) * RES
    zc = Z_RANGE[0] + (rows + 0.5) * RES

    y_top = _denorm(y_min_n)   # highest point (camera y down)
    y_low = _denorm(y_max_n)   # lowest point

    plane, n_inliers = fit_ground_plane(xc[observed], zc[observed], y_low[observed])
    a, b, c = plane
    y_ground = a * xc + b * zc + c

    h_top = np.where(observed, y_ground - y_top, np.nan)
    h_bot = np.where(observed, y_ground - y_low, np.nan)

    with np.errstate(invalid="ignore"):
        sticks_up = (h_top > min_obstacle_height) & (h_bot < max_obstacle_height)
        blocks_agent = h_bot < agent_height
    obstacle = observed & sticks_up & blocks_agent

    n_see_through = 0
    if see_through_check and obstacle.any():
        # A cell whose LOWEST point lies on the road proves the camera saw that road,
        # whatever junk sits above it in the same cell.
        with np.errstate(invalid="ignore"):
            road_seen = observed & (np.abs(h_bot) < 0.15)
        obstacle, n_see_through = remove_see_through(
            obstacle, road_seen, xc, zc, h_top, cam_h=c,
            min_obstacle_height=min_obstacle_height, agent_height=agent_height)

    # Stereo depth error grows ~ range^2, so far-away noise forms bigger specks.
    # Drop blobs smaller than min_blob_cells + blob_range_coef * r^2 (r = blob's nearest range).
    # A pedestrian at 30 m still covers ~50 cells; this threshold is 22 cells there.
    # Far objects reach the grid as scattered fragments, so far fragments closer than
    # 2 * blob_link_cells cells are grouped (dilated mask) before their size is judged.
    # Near range is dense enough; linking there would merge isolated noise specks.
    if min_blob_cells > 1 and obstacle.any():
        linked = obstacle
        if blob_link_cells:
            far = obstacle & (np.hypot(xc, zc) > blob_link_min_range)
            linked = obstacle | ndimage.binary_dilation(far, iterations=blob_link_cells)
        lab, n = ndimage.label(linked, structure=np.ones((3, 3), bool))
        lab = np.where(obstacle, lab, 0)
        idx = np.arange(1, n + 1)
        sizes = ndimage.sum(obstacle, lab, index=idx)
        near = ndimage.minimum(np.hypot(xc, zc), lab, index=idx)
        thresh = min_blob_cells + blob_range_coef * near ** 2
        small = np.isin(lab, idx[sizes < thresh])
        obstacle &= ~small

    state = np.full((H, W), UNKNOWN, np.uint8)
    state[observed] = FREE
    state[obstacle] = OBSTACLE

    return OccupancyGrid(
        state=state, height_top=h_top, height_bottom=h_bot, ground_plane=plane,
        meta=dict(observed_frac=float(observed.mean()),
                  obstacle_frac=float(obstacle.mean()),
                  see_through_removed=n_see_through,
                  ground_inliers=n_inliers,
                  camera_height_est=float(c),
                  min_obstacle_height=min_obstacle_height,
                  agent_height=agent_height),
    )
