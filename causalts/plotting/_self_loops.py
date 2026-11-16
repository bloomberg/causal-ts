"""Self-loop glyphs shared by ``compare_graphs`` and ``plot_graph``.

A lagged self-link ``X(t-tau) -> X(t)`` is drawn as a small loop on the
node, pointing into the widest gap between the node's other edges so it does
not cross them. Several loops on one node fan out side by side in that gap.
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path


def _self_loop_angle(i, pos, edges):
    """Centre and width (radians) of the widest gap between node *i*'s edges.

    *edges* holds ``(a, b, rad)`` for every drawn ``arc3`` edge. An arc leaves
    and enters its nodes along the tangent towards its Bezier control point,
    not along the straight chord, so the gap is measured from those tangents.
    """
    angles = []
    for a, b, rad in edges:
        if i not in (a, b) or a == b:
            continue
        (xa, ya), (xb, yb) = pos[a], pos[b]
        # arc3 control point: chord midpoint shifted by rad * the chord
        # rotated clockwise (matplotlib's ConnectionStyle.Arc3).
        cx = (xa + xb) / 2 + rad * (yb - ya)
        cy = (ya + yb) / 2 - rad * (xb - xa)
        x0, y0 = pos[i]
        angles.append(np.arctan2(cy - y0, cx - x0))
    angles.sort()
    if not angles:
        return np.pi / 2, 2 * np.pi
    gaps = np.diff(angles + [angles[0] + 2 * np.pi])
    k = int(np.argmax(gaps))
    return angles[k] + gaps[k] / 2, gaps[k]


_END_MARKERS = {"o": "o", "x": "x", "+": "+"}

_LOOP_HEADS = {
    True: "-|>,head_length={hl},head_width={hw}",
    False: "-",
    "both": "<|-|>,head_length={hl},head_width={hw}",
}


def _ray_exit(radii, phi):
    """Distance along direction *phi* to an ellipse with semi-axes *radii*."""
    rx, ry = radii
    return 1.0 / np.hypot(np.cos(phi) / rx, np.sin(phi) / ry)


def draw_loops(
    ax,
    centre_xy,
    node_size,
    gap,
    loops,
    draw_label=None,
    node_width=None,
    transform=None,
    alpha=None,
):
    """Draw *loops* on the node at *centre_xy*.

    *gap* is ``(direction, width)`` from ``_self_loop_angle``. Each loop is
    ``(color, linestyle, linewidth, label, arrow, zorder[, ends])``; *arrow*
    is True (head at the end), False (no head) or ``"both"``; *ends* is an
    optional ``(tail, head)`` pair of endpoint symbols, ``"o"`` (circle),
    ``"x"`` or ``"+"``, drawn near the loop's ends. *alpha* None keeps the
    colours' own opacity. *draw_label* is
    called as ``draw_label(ax, xy, label)``, or labels are skipped if None.
    *node_size* is the node height; *node_width* (default: the height) lets
    loops start and end on an elliptical node's boundary. *transform* is the
    coordinate system of *centre_xy* (default ``ax.transData``).
    """
    centre, width = gap
    radii = np.array([node_width if node_width is not None else node_size, node_size])
    radii = radii / 2
    step = min(np.deg2rad(80), width / len(loops))
    # Half-angle at the node; the loop body fans out to about 1.6x this.
    spread = min(np.deg2rad(30), step * 0.3)
    c = np.asarray(centre_xy, dtype=float)

    def unit(phi):
        return np.array([np.cos(phi), np.sin(phi)])

    for k, (color, linestyle, lw, label, arrow, zorder, *ends) in enumerate(loops):
        theta = centre + (k - (len(loops) - 1) / 2) * step
        # Cubic loop leaving and re-entering the node either side of theta.
        # Reach past the node in this direction: on an elongated ellipse the
        # boundary along theta is further out than the height radius.
        reach = node_size * 1.3 + (_ray_exit(radii, theta) - radii[1])
        # Where the rays at theta +- spread leave the (elliptical) node, so
        # the endpoints keep the angles chosen for the gap.
        p_start = c + _ray_exit(radii, theta + spread) * unit(theta + spread)
        p_end = c + _ray_exit(radii, theta - spread) * unit(theta - spread)
        c1 = p_start + reach * unit(theta + 1.6 * spread)
        c2 = p_end + reach * unit(theta - 1.6 * spread)
        path = Path(
            [p_start, c1, c2, p_end],
            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
        )
        loop = FancyArrowPatch(
            path=path,
            arrowstyle=_LOOP_HEADS[arrow].format(
                hl=0.3 + 0.05 * lw, hw=0.15 + 0.03 * lw
            ),
            mutation_scale=10,
            lw=lw,
            linestyle=linestyle,
            color=color,
            clip_on=False,
            zorder=zorder,
            capstyle="butt",
            transform=ax.transData if transform is None else transform,
            alpha=alpha,
        )
        loop.set_gid("self-loop")
        ax.add_artist(loop)
        tail, head = ends[0] if ends else (None, None)
        for t, symbol in ((0.12, tail), (0.88, head)):
            if symbol not in _END_MARKERS:
                continue
            xy = (
                (1 - t) ** 3 * p_start
                + 3 * (1 - t) ** 2 * t * c1
                + 3 * (1 - t) * t**2 * c2
                + t**3 * p_end
            )
            ax.plot(
                *xy,
                marker=_END_MARKERS[symbol],
                markersize=3 + lw,
                markerfacecolor="white" if symbol == "o" else color,
                markeredgecolor=color,
                markeredgewidth=max(lw * 0.6, 0.8),
                linestyle="none",
                clip_on=False,
                zorder=zorder + 0.5,
                transform=ax.transData if transform is None else transform,
                alpha=alpha,
                gid="self-loop-end",
            )
        if draw_label is not None and label:
            apex = (p_start + 3 * c1 + 3 * c2 + p_end) / 8
            draw_label(ax, apex + 0.35 * node_size * unit(theta), label)
