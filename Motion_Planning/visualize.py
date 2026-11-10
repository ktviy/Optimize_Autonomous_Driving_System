import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from bev_occupancy import OBSTACLE, UNKNOWN

MODE_COLORS = ["#e6194b", "#f58231", "#ffe119"]


def draw_result(ax, res, title=None, zoom=None):
    grid = res["grid"]
    ext = grid.extent

    # background: height above ground on observed cells, grey for unknown
    h = np.clip(np.nan_to_num(grid.height_top, nan=0.0), 0, 3)
    ax.imshow(np.where(grid.state == UNKNOWN, np.nan, h), origin="lower", extent=ext,
              cmap="Greens_r", vmin=-1.5, vmax=3, interpolation="nearest")
    ax.imshow(np.where(grid.state == UNKNOWN, 1.0, np.nan), origin="lower", extent=ext,
              cmap=ListedColormap(["#d9d9d9"]), interpolation="nearest")
    ax.imshow(np.where(grid.state == OBSTACLE, 1.0, np.nan), origin="lower", extent=ext,
              cmap=ListedColormap(["#222222"]), interpolation="nearest")

    for c in res["candidates"][:: max(1, len(res["candidates"]) // 120)]:
        ax.plot(c[:, 0], c[:, 1], color="#4a90d9", lw=0.4, alpha=0.35)

    path = res["route"]["path"]
    if len(path):
        ax.plot(path[:, 0], path[:, 1], "--", color="#00bcd4", lw=1.5, label="global route")
    gx, gz = res["route"]["goal"]
    ax.plot(gx, gz, "*", color="#00bcd4", ms=14, mec="k", label="goal")

    offsets, radius = res["footprint"]
    for k, m in reversed(list(enumerate(res["modes"]))):
        s = m["states"]
        col = MODE_COLORS[k % len(MODE_COLORS)]
        lbl = f"mode {k + 1}: p={m['prob']:.2f}" if not res["emergency_stop"] else "EMERGENCY STOP"
        ax.plot(s[:, 0], s[:, 1], color=col, lw=2.6 if k == 0 else 1.8, label=lbl, zorder=5)
        ax.plot(s[::10, 0], s[::10, 1], "o", color=col, ms=3, zorder=6)  # 1 s ticks
        if k == 0:
            x, z, psi = s[-1, 0], s[-1, 1], s[-1, 2]
            for o in offsets:
                ax.add_patch(plt.Circle((x + o * np.sin(psi), z + o * np.cos(psi)), radius,
                                        fill=False, ec=col, lw=1.2, zorder=6))
    s0 = res["modes"][0]["states"][0]
    for o in offsets:
        ax.add_patch(plt.Circle((s0[0] + o * np.sin(s0[2]), s0[1] + o * np.cos(s0[2])), radius,
                                fill=False, ec="#2ca02c", lw=1.5, zorder=6))
    ax.plot(0, 0, "^", color="#2ca02c", ms=9, mec="k", zorder=7, label="camera / start")

    if zoom:
        ax.set_xlim(*zoom[0])
        ax.set_ylim(*zoom[1])
    else:
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(-3, ext[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x lateral (m)")
    ax.set_ylabel("z forward (m)")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.85)


def save_figure(results, frame_id, out_path):
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 9))
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, results):
        best = res["modes"][0]
        v_end = best["states"][-1, 3]
        title = (f"{frame_id} | {res['agent']} | v0={res['v0']:.1f} m/s -> {v_end:.1f} m/s\n"
                 f"feasible {res['n_feasible']}/{res['n_candidates']} rollouts"
                 + (" | NO SAFE PATH: braking" if res["emergency_stop"] else ""))
        zoom = None
        if res["agent"] == "robot":
            zoom = ((-7, 7), (-2, 16))
        draw_result(ax, res, title, zoom)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
