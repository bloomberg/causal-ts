"""
This file has been modified by Mohammad Fesanghary on July 28 2025.

Original Author: Jakob Runge <jakob@jakob-runge.com>
Original Package: tigramite
Original License: GNU General Public License v3.0

This module provides functions and classes for visualizing time series data,
lag functions, scatter plots, density plots, and networks. It is designed
to work with the Tigramite framework for causal inference in time series data.

Features:
- Time series plotting with options for masked samples and mean lines.
- Lag function visualization with customizable matrix setup.
- Scatter plot matrix for pairwise variable relationships.
- Density plot matrix for marginal and joint distributions.
- Network visualization with curved edges and node attributes.

Note:
This code is part of the Tigramite framework and adheres to the GNU General
Public License v3.0. Ensure proper attribution when using or modifying this code.
"""

# SPDX-License-Identifier: GPL-3.0-or-later

from copy import deepcopy
from operator import sub
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import matplotlib.transforms as transforms

# import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot, ticker


def _myround(x, base=5, round_mode="updown"):
    """Rounds x to a float with precision base."""

    if round_mode == "updown":
        return base * round(float(x) / base)
    elif round_mode == "down":
        return base * np.floor(float(x) / base)
    elif round_mode == "up":
        return base * np.ceil(float(x) / base)

    return base * round(float(x) / base)


def _draw_network_with_curved_edges(
    fig,
    ax,
    G,
    pos,
    node_rings,
    node_labels,
    node_label_size=10,
    standard_size=100,
    node_aspect=None,
    standard_cmap="OrRd",
    standard_color_links="black",
    standard_color_nodes="lightgrey",
    cmap_links="YlOrRd",
    links_vmin=0.0,
    links_vmax=1.0,
    links_ticks=0.2,
    link_label_fontsize=8,
    curved_radius=0.2,
    label_fontsize=4,
    link_colorbar_label="link",
    tick_label_size=6,
    inner_edge_curved=False,
    show_colorbar=True,
    special_nodes=None,
    autodep_sig_lags=None,
    show_autodependency_lags=False,
    transform="data",
    node_classification=None,
    max_lag=0,
):
    """Function to draw a network from networkx graph instance.
    Various attributes are used to specify the graph's properties.
    This function is just a beta-template for now that can be further
    customized.
    """

    if transform == "data":
        transform = ax.transData

    from matplotlib.patches import Ellipse, FancyArrowPatch

    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.set_xticks([])
    ax.set_yticks([])

    N = len(G)

    # This fixes a positioning bug in matplotlib.
    ax.scatter(0, 0, zorder=-10, alpha=0)

    def draw_edge(
        ax,
        u,
        v,
        d,
        seen,
        arrowstyle="Simple, head_width=2, head_length=2, tail_width=1",
        outer_edge=True,
    ):
        # avoiding attribute error raised by changes in networkx
        if hasattr(G, "node"):
            # works with networkx 1.10
            n1 = G.node[u]["patch"]
            n2 = G.node[v]["patch"]
        else:
            # works with networkx 2.4
            n1 = G.nodes[u]["patch"]
            n2 = G.nodes[v]["patch"]

        # print("+++++++++++++++++++++++==cmap_links ", cmap_links)
        if outer_edge:
            rad = -1.0 * curved_radius
            if cmap_links is not None:
                facecolor = data_to_rgb_links.to_rgba(d["outer_edge_color"])
            else:
                if d["outer_edge_color"] is not None:
                    facecolor = d["outer_edge_color"]
                else:
                    facecolor = standard_color_links

            width = d["outer_edge_width"]
            alpha = d["outer_edge_alpha"]
            if (u, v) in seen:
                rad = seen.get((u, v))
                rad = (rad + np.sign(rad) * 0.1) * -1.0
            arrowstyle = arrowstyle
            # link_edge = d['outer_edge_edge']
            linestyle = "solid"  # d.get("outer_edge_style")

            if d.get("outer_edge_attribute", None) == "spurious":
                facecolor = "grey"

            if d.get("outer_edge_type") in ["<-o", "<--", "<-x", "<-+"]:
                n1, n2 = n2, n1

            if d.get("outer_edge_type") in [
                "o-o",
                "o--",
                "--o",
                "---",
                "x-x",
                "x--",
                "--x",
                "o-x",
                "x-o",
                # "+->",
                # "<-+",
            ]:
                arrowstyle = "-"
                # linewidth = width*factor
            elif d.get("outer_edge_type") == "<->":
                # arrowstyle = "<->, head_width=0.4, head_length=1"
                arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1"
            elif d.get("outer_edge_type") in [
                "o->",
                "-->",
                "<-o",
                "<--",
                "<-x",
                "x->",
                "+->",
                "<-+",
            ]:
                # arrowstyle = "->, head_width=0.4, head_length=1"
                # arrowstyle = "->, head_width=0.4, head_length=1, width=10"
                arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1"
            else:
                arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1"
                # raise ValueError("edge type %s not valid." %d.get("outer_edge_type"))
        else:
            rad = -1.0 * inner_edge_curved * curved_radius
            if cmap_links is not None:
                facecolor = data_to_rgb_links.to_rgba(d["inner_edge_color"])
            else:
                if d["inner_edge_color"] is not None:
                    facecolor = d["inner_edge_color"]
                else:
                    # print("HERE")
                    facecolor = standard_color_links

            width = d["inner_edge_width"]
            alpha = d["inner_edge_alpha"]

            if d.get("inner_edge_attribute", None) == "spurious":
                facecolor = "grey"
            # print(d.get("inner_edge_type"))
            if d.get("inner_edge_type") in ["<-o", "<--", "<-x", "<-+"]:
                n1, n2 = n2, n1

            if d.get("inner_edge_type") in [
                "o-o",
                "o--",
                "--o",
                "---",
                "x-x",
                "x--",
                "--x",
                "o-x",
                "x-o",
            ]:
                arrowstyle = "-"
            elif d.get("inner_edge_type") == "<->":
                # arrowstyle = "<->, head_width=0.4, head_length=1"
                arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1"
            elif d.get("inner_edge_type") in [
                "o->",
                "-->",
                "<-o",
                "<--",
                "<-x",
                "x->",
                "+->",
                "<-+",
            ]:
                # arrowstyle = "->, head_width=0.4, head_length=1"
                arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1"
            else:
                arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1"

            #     raise ValueError("edge type %s not valid." %d.get("inner_edge_type"))

            linestyle = "solid"  # d.get("inner_edge_style")

        coor1 = n1.center
        coor2 = n2.center

        marker_size = width**2
        # figuresize = fig.get_size_inches()

        if (outer_edge is True and d.get("outer_edge_type") == "<->") or (
            outer_edge is False and d.get("inner_edge_type") == "<->"
        ):
            e_p = FancyArrowPatch(
                coor1,
                coor2,
                arrowstyle=arrowstyle,
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=1 * width,
                lw=0.0,  # width / 2.,
                aa=True,
                alpha=alpha,
                linestyle=linestyle,
                color=facecolor,
                clip_on=False,
                patchA=n1,
                patchB=n2,
                shrinkA=7,
                shrinkB=0,
                zorder=-1,
                capstyle="butt",
                transform=transform,
            )
            ax.add_artist(e_p)

            e_p_back = FancyArrowPatch(
                coor2,
                coor1,
                arrowstyle=arrowstyle,
                connectionstyle=f"arc3,rad={-rad}",
                mutation_scale=1 * width,
                lw=0.0,  # width / 2.,
                aa=True,
                alpha=alpha,
                linestyle=linestyle,
                color=facecolor,
                clip_on=False,
                patchA=n2,
                patchB=n1,
                shrinkA=7,
                shrinkB=0,
                zorder=-1,
                capstyle="butt",
                transform=transform,
            )
            ax.add_artist(e_p_back)

        else:
            if arrowstyle == "-":
                lw = 1 * width
            else:
                lw = 0.0

            e_p = FancyArrowPatch(
                coor1,
                coor2,
                arrowstyle=arrowstyle,
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=1 * width,
                lw=lw,  # width / 2.,
                aa=True,
                alpha=alpha,
                linestyle=linestyle,
                color=facecolor,
                clip_on=False,
                patchA=n1,
                patchB=n2,
                shrinkA=0,
                shrinkB=0,
                # zorder=-1,
                capstyle="butt",
                transform=transform,
            )
            ax.add_artist(e_p)

        e_p_marker = FancyArrowPatch(
            coor1,
            coor2,
            arrowstyle="-",
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=1 * width,
            lw=0.0,  # width / 2.,
            aa=True,
            alpha=0.0,
            linestyle=linestyle,
            color=facecolor,
            clip_on=False,
            patchA=n1,
            patchB=n2,
            shrinkA=0,
            shrinkB=0,
            zorder=-10,
            capstyle="butt",
            transform=transform,
        )
        ax.add_artist(e_p_marker)

        # marker_path = e_p_marker.get_path()
        vertices = e_p_marker.get_path().vertices.copy()
        # vertices = e_p_marker.get_verts()
        # vertices = e_p_marker.get_path().to_polygons(transform=None)[0]
        # print(vertices.shape)
        # m, n = vertices.shape

        # print(vertices)
        start = vertices[0]
        end = vertices[-1]

        # This must be added to avoid rescaling of the plot, when no 'o'
        # or 'x' is added to the graph.
        ax.scatter(
            *start,
            zorder=-10,
            alpha=0,
            transform=transform,
        )

        if outer_edge:
            if d.get("outer_edge_type") in ["o->", "o--"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("outer_edge_type") == "<-o":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "--o":
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") in ["x--", "x->"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("outer_edge_type") in ["+--", "+->"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="P",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("outer_edge_type") == "<-x":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "<-+":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="P",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "--x":
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "o-o":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "x-x":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "o-x":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("outer_edge_type") == "x-o":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)

        else:
            if d.get("inner_edge_type") in ["o->", "o--"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("inner_edge_type") == "<-o":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "--o":
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") in ["x--", "x->"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("inner_edge_type") in ["+--", "+->"]:
                circle_marker_start = ax.scatter(
                    *start,
                    marker="P",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
            elif d.get("inner_edge_type") == "<-x":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "<-+":
                circle_marker_end = ax.scatter(
                    *start,
                    marker="P",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "--x":
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "o-o":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "x-x":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "o-x":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)
            elif d.get("inner_edge_type") == "x-o":
                circle_marker_start = ax.scatter(
                    *start,
                    marker="X",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_start)
                circle_marker_end = ax.scatter(
                    *end,
                    marker="o",
                    s=marker_size,
                    facecolor="w",
                    edgecolor=facecolor,
                    zorder=1,
                    transform=transform,
                )
                ax.add_collection(circle_marker_end)

        if d["label"] is not None and outer_edge:

            def closest_node(node, nodes):
                nodes = np.asarray(nodes)
                node = node.reshape(1, 2)
                dist_2 = np.sum((nodes - node) ** 2, axis=1)
                return np.argmin(dist_2)

            # Attach labels of lags
            # trans = None  # patch.get_transform()
            # path = e_p.get_path()
            vertices = e_p_marker.get_path().vertices.copy()
            verts = e_p.get_path().to_polygons(transform=None)[0]
            # print(verts)
            # print(verts.shape)
            # print(vertices.shape)
            # for num, vert in enumerate(verts):
            #     ax.text(vert[0], vert[1], str(num),
            #         transform=transform,)
            # ax.scatter(verts[:,0], verts[:,1])
            # mid_point = np.array([(start[0] + end[0])/2., (start[1] + end[1])/2.])
            # print(start, end, mid_point)
            # ax.scatter(mid_point[0], mid_point[1], marker='x',
            #     s=100, zorder=10, transform=transform,)
            closest_node = closest_node(vertices[int(len(vertices) / 2.0), :], verts)
            # print(closest_node, verts[closest_node])
            # ax.scatter(verts[closest_node][0], verts[closest_node][1], marker='x')

            if len(vertices) > 2:
                # label_vert = vertices[int(len(vertices)/2.),:] #verts[1, :]
                label_vert = verts[closest_node]  # verts[1, :]
                ll = d["label"]
                string = str(ll)
                txt = ax.text(
                    label_vert[0],
                    label_vert[1],
                    string,
                    fontsize=link_label_fontsize,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="w",
                    zorder=1,
                    transform=transform,
                )
                txt.set_path_effects(
                    [PathEffects.withStroke(linewidth=2, foreground="k")]
                )

        return rad

    # Collect all edge weights to get color scale
    all_links_weights = []
    # all_links_edge_weights = []
    for u, v, d in G.edges(data=True):
        if u != v:
            if d["outer_edge"] and d["outer_edge_color"] is not None:
                all_links_weights.append(d["outer_edge_color"])
            if d["inner_edge"] and d["inner_edge_color"] is not None:
                all_links_weights.append(d["inner_edge_color"])

    if cmap_links is not None and len(all_links_weights) > 0:
        if links_vmin is None:
            links_vmin = np.array(all_links_weights).min()
        if links_vmax is None:
            links_vmax = np.array(all_links_weights).max()
        data_to_rgb_links = pyplot.cm.ScalarMappable(
            norm=None, cmap=pyplot.get_cmap(cmap_links)
        )
        data_to_rgb_links.set_array(np.array(all_links_weights))
        data_to_rgb_links.set_clim(vmin=links_vmin, vmax=links_vmax)
        # Create colorbars for links

        # setup colorbar axes.
        if show_colorbar:
            # bbox_ax = ax.get_position()
            cax_e = ax.inset_axes(
                [
                    0.55,
                    -0.07,
                    0.4,
                    0.07,
                    # bbox_ax.xmax - width*0.45,
                    # bbox_ax.ymin-0.075*height+network_lower_bound-0.15,
                    # width*0.4,
                    # 0.075*height,   #0.025 + (len(all_links_edge_weights) == 0) * 0.035,
                ],
                frameon=False,
            )
            # divider = make_axes_locatable(ax)

            # cax_e = divider.append_axes('bottom', size='5%', pad=0.05, frameon=False,)

            cb_e = pyplot.colorbar(
                data_to_rgb_links, cax=cax_e, orientation="horizontal"
            )
            # try:
            ticks_here = np.arange(
                _myround(links_vmin, links_ticks, "down"),
                _myround(links_vmax, links_ticks, "up") + links_ticks,
                links_ticks,
            )
            cb_e.set_ticks(
                ticks_here[(links_vmin <= ticks_here) & (ticks_here <= links_vmax)]
            )
            # except:
            #     print('no ticks given')

            cb_e.outline.clear()
            cax_e.set_xlabel(
                link_colorbar_label, labelpad=1, fontsize=label_fontsize, zorder=10
            )
            cax_e.tick_params(axis="both", which="major", labelsize=tick_label_size)

    ##
    # Draw nodes
    ##
    node_sizes = np.zeros((len(node_rings), N))
    for ring in list(node_rings):  # iterate through to get all node sizes
        if node_rings[ring]["sizes"] is not None:
            node_sizes[ring] = node_rings[ring]["sizes"]

        else:
            node_sizes[ring] = standard_size
    # max_sizes = node_sizes.max(axis=1)
    total_max_size = node_sizes.sum(axis=0).max()
    node_sizes /= total_max_size
    node_sizes *= standard_size

    def get_aspect(ax):
        # Total figure size
        figW, figH = ax.get_figure().get_size_inches()
        # print(figW, figH)
        # Axis size on figure
        _, _, w, h = ax.get_position().bounds
        # Ratio of display units
        # print(w, h)
        disp_ratio = (figH * h) / (figW * w)
        # Ratio of data units
        # Negative over negative because of the order of subtraction
        data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
        # print(data_ratio, disp_ratio)
        return disp_ratio / data_ratio

    if node_aspect is None:
        node_aspect = 1.0

    # start drawing the outer ring first...
    for ring in list(node_rings)[::-1]:
        #        print ring
        # dictionary of rings: {0:{'sizes':(N,)-array, 'color_array':(N,)-array
        # or None, 'cmap':string, 'vmin':float or None, 'vmax':float or None}}
        if node_rings[ring]["color_array"] is not None:
            color_data = node_rings[ring]["color_array"]
            if node_rings[ring]["vmin"] is not None:
                vmin = node_rings[ring]["vmin"]
            else:
                vmin = node_rings[ring]["color_array"].min()
            if node_rings[ring]["vmax"] is not None:
                vmax = node_rings[ring]["vmax"]
            else:
                vmax = node_rings[ring]["color_array"].max()
            if node_rings[ring]["cmap"] is not None:
                cmap = node_rings[ring]["cmap"]
            else:
                cmap = standard_cmap
            data_to_rgb = pyplot.cm.ScalarMappable(
                norm=None, cmap=pyplot.get_cmap(cmap)
            )
            data_to_rgb.set_array(color_data)
            data_to_rgb.set_clim(vmin=vmin, vmax=vmax)
            colors = [data_to_rgb.to_rgba(color_data[n]) for n in G]

            if node_rings[ring]["colorbar"]:
                # Create colorbars for nodes
                # cax_n = pyplot.axes([.8 + ring*0.11,
                # ax.get_subplotspec().get_position(ax.figure).bounds[1]+0.05, 0.025, 0.35], frameon=False) #
                # setup colorbar axes.
                # setup colorbar axes.
                # bbox_ax = ax.get_position()
                # print(bbox_ax.xmin, bbox_ax.xmax, bbox_ax.ymin, bbox_ax.ymax)
                cax_n = ax.inset_axes(
                    [
                        0.05,
                        -0.07,
                        0.4,
                        0.07,
                        # bbox_ax.xmin + width*0.05,
                        # bbox_ax.ymin-0.075*height+network_lower_bound-0.15,
                        # width*0.4,
                        # 0.075*height,   #0.025 + (len(all_links_edge_weights) == 0) * 0.035,
                    ],
                    frameon=False,
                )
                cb_n = pyplot.colorbar(data_to_rgb, cax=cax_n, orientation="horizontal")
                # try:
                ticks_here = np.arange(
                    _myround(vmin, node_rings[ring]["ticks"], "down"),
                    _myround(vmax, node_rings[ring]["ticks"], "up")
                    + node_rings[ring]["ticks"],
                    node_rings[ring]["ticks"],
                )
                cb_n.set_ticks(ticks_here[(vmin <= ticks_here) & (ticks_here <= vmax)])
                # except:
                #     print ('no ticks given')
                cb_n.outline.clear()
                # cb_n.set_ticks()
                cax_n.set_xlabel(
                    node_rings[ring]["label"], labelpad=1, fontsize=label_fontsize
                )
                cax_n.tick_params(axis="both", which="major", labelsize=tick_label_size)
        else:
            colors = None
            vmin = None
            vmax = None

        for n in G:
            # if type(node_alpha) == dict:
            #     alpha = node_alpha[n]
            # else:
            #     alpha = 1.0

            if special_nodes is not None:
                if n in special_nodes:
                    color_here = special_nodes[n]
                else:
                    color_here = standard_color_nodes
            else:
                if colors is None:
                    color_here = standard_color_nodes
                else:
                    color_here = colors[n]

            c = Ellipse(
                pos[n],
                width=node_sizes[: ring + 1].sum(axis=0)[n] * node_aspect,
                height=node_sizes[: ring + 1].sum(axis=0)[n],
                clip_on=False,
                facecolor=color_here,
                edgecolor=color_here,
                zorder=-ring - 1 + 2,
                transform=transform,
            )

            # else:
            #     if special_nodes is not None and n in special_nodes:
            #         color_here = special_nodes[n]
            #     else:
            #         color_here = colors[n]
            #     c = Ellipse(
            #         pos[n],
            #         width=node_sizes[: ring + 1].sum(axis=0)[n] * node_aspect,
            #         height=node_sizes[: ring + 1].sum(axis=0)[n],
            #         clip_on=False,
            #         facecolor=colors[n],
            #         edgecolor=colors[n],
            #         zorder=-ring - 1,
            #     )

            ax.add_patch(c)

            if node_classification is not None and node_classification[n] in [
                "space_context_last",
                "space_dummy_last",
                "time_dummy_last",
            ]:
                node_height = node_sizes[: ring + 1].sum(axis=0)[n]
                node_width_difference_to_height = node_height * (1 - node_aspect)

                c_wide = mpatches.FancyBboxPatch(
                    (
                        pos[n - max_lag + 1][0] + node_width_difference_to_height / 2,
                        pos[n - max_lag + 1][1],
                    ),
                    (
                        pos[n][0]
                        - pos[n - max_lag + 1][0]
                        - node_width_difference_to_height
                    ),
                    0.0,
                    boxstyle=mpatches.BoxStyle.Round(pad=0.5 * node_height),
                    facecolor=color_here,
                    edgecolor=color_here,
                )

                ax.add_patch(c_wide)

            # avoiding attribute error raised by changes in networkx
            if hasattr(G, "node"):
                # works with networkx 1.10
                G.node[n]["patch"] = c
            else:
                # works with networkx 2.4
                G.nodes[n]["patch"] = c

            if ring == 0:
                ax.text(
                    pos[n][0],
                    pos[n][1],
                    node_labels[n],
                    fontsize=node_label_size,
                    horizontalalignment="center",
                    verticalalignment="center",
                    alpha=1.0,
                    zorder=5.0,
                    transform=transform,
                )
                if show_autodependency_lags:
                    ax.text(
                        pos[n][0],
                        pos[n][1],
                        autodep_sig_lags[n],
                        fontsize=link_label_fontsize,
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="black",
                        zorder=5.0,
                        transform=transform,
                    )

    # Draw edges
    seen = {}
    for u, v, d in G.edges(data=True):
        if d.get("no_links"):
            d["inner_edge_alpha"] = 1e-8
            d["outer_edge_alpha"] = 1e-8
        if u != v:
            if d["outer_edge"]:
                seen[(u, v)] = draw_edge(ax, u, v, d, seen, outer_edge=True)
            if d["inner_edge"]:
                seen[(u, v)] = draw_edge(ax, u, v, d, seen, outer_edge=False)

    ax.set_aspect("equal")


# ---------- normalization helper ----------


def normalize_positions_to_unit_square(
    pos_dict: Dict[Any, Any],
    target_half_size: float = 1.0,
    keep_aspect: bool = True,
    margin: float = 0.0,
    flip_y: bool = False,
) -> Dict[Any, Any]:
    nodes = list(pos_dict.keys())
    pts = np.array([pos_dict[n] for n in nodes], dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pos_dict must map nodes to 2D coordinates (x, y).")

    xy_min = pts.min(axis=0)
    xy_max = pts.max(axis=0)
    center = 0.5 * (xy_min + xy_max)
    pts_centered = pts - center

    span = np.maximum(xy_max - xy_min, 1e-12)
    usable = 2.0 * target_half_size * (1.0 - margin)

    if keep_aspect:
        s = usable / max(span)
        pts_scaled = pts_centered * s
    else:
        sx, sy = usable / span[0], usable / span[1]
        pts_scaled = pts_centered * np.array([sx, sy])

    if flip_y:
        pts_scaled[:, 1] *= -1.0

    pts_scaled = np.clip(pts_scaled, -target_half_size, target_half_size)
    return {n: (float(p[0]), float(p[1])) for n, p in zip(nodes, pts_scaled)}


# ---------- KK enhancements ----------


def kk_positions_packed(
    G: nx.Graph, weight="weight", scale: float = 1.0, seed=None, pad: float = 2.5
):
    """
    Lay out each connected component with KK, then pack components on a grid to avoid overlap.
    pad: spacing multiplier between component boxes.
    """
    G_und = G.to_undirected()
    comps = [G.subgraph(c).copy() for c in nx.connected_components(G_und)]
    rng = np.random.default_rng(seed)  # noqa: F841

    comp_pos, comp_boxes = [], []
    for H in comps:
        pos = nx.kamada_kawai_layout(H, weight=weight, scale=scale)
        xs = np.array([p[0] for p in pos.values()])
        ys = np.array([p[1] for p in pos.values()])
        box = (xs.min(), ys.min(), xs.max(), ys.max())
        comp_pos.append(pos)
        comp_boxes.append(box)

    cols = int(np.ceil(np.sqrt(len(comps))))
    out = {}
    # typical component size for spacing
    widths = [bx[2] - bx[0] for bx in comp_boxes]
    heights = [by[3] - by[1] for by in comp_boxes]
    w = (np.median(widths) if widths else 1.0) * pad
    h = (np.median(heights) if heights else 1.0) * pad

    for idx, (H, pos, box) in enumerate(zip(comps, comp_pos, comp_boxes)):
        r, c = divmod(idx, cols)
        ox, oy = c * w, -r * h  # grid offsets
        minx, miny, _, _ = box
        for n, (x, y) in pos.items():
            out[n] = (x - minx + ox, y - miny + oy)
    return out


def spread_duplicates(pos: Dict[Any, Any], min_sep: float = 1e-2, seed=None):
    """Nudge nodes that share identical (x,y) onto a small ring."""
    rng = np.random.default_rng(seed)  # noqa: F841
    buckets: Dict[tuple, list] = {}
    for n, (x, y) in pos.items():
        key = (round(x, 8), round(y, 8))
        buckets.setdefault(key, []).append(n)

    new_pos = dict(pos)
    for nodes in buckets.values():
        k = len(nodes)
        if k <= 1:
            continue
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        for a, n in zip(angles, nodes):
            dx = min_sep * np.cos(a)
            dy = min_sep * np.sin(a)
            x, y = new_pos[n]
            new_pos[n] = (x + dx, y + dy)
    return new_pos


def relax_with_spring(
    G: nx.Graph,
    pos: Dict[Any, Any],
    iterations: int = 5,
    k: Optional[float] = None,
    seed=None,
):
    """Short spring pass from KK positions to break ties."""
    return nx.spring_layout(
        G, pos=pos, fixed=None, iterations=iterations, k=k, seed=seed
    )


# ---------- per-layout defaults & utils ----------


def _auto_shells(G: nx.Graph, max_shells: int = 3) -> List[List[Any]]:
    if G.number_of_nodes() == 0:
        return [[]]
    deg = dict(G.degree())
    if len(deg) <= max_shells:
        nodes = sorted(deg, key=deg.get, reverse=True)
        return [[n] for n in nodes]
    dvals = np.array(sorted(deg.values()))
    q = np.quantile(dvals, [0.33, 0.66]) if len(dvals) > 2 else dvals.mean()
    inner, mid, outer = [], [], []
    for n, v in deg.items():
        if v >= (q[1] if isinstance(q, np.ndarray) else q):
            inner.append(n)
        elif isinstance(q, np.ndarray) and v >= q[0]:
            mid.append(n)
        else:
            outer.append(n)
    shells = [s for s in [inner, mid, outer] if len(s)]
    return shells if shells else [list(G.nodes())]


def _default_kwargs_for_layout(
    G: nx.Graph, layout: str, seed: Optional[int]
) -> Dict[str, Any]:
    name = layout.lower()
    n = max(G.number_of_nodes(), 1)
    if name in ("spring", "fr", "fruchterman_reingold"):
        return {
            "k": 1.0 / np.sqrt(n),
            "iterations": 100 if n <= 200 else 50,
            "seed": seed,
        }
    if name in ("kamada_kawai", "kk"):
        return {"weight": "weight"}
    if name in ("spectral", "circular", "shell", "random", "spiral", "planar"):
        return {"center": None, "scale": 1.0}
    if name in ("kk_enhanced",):
        return {
            "weight": "weight",
            # "scale": 1.0,
            # "pad": 2.5,
            "min_sep": 1e-2,
            "spring_iterations": 1,
            "spring_k": None,
            "seed": seed,
        }
    if name in ("graphviz", "neato", "dot", "fdp", "sfdp", "twopi", "circo"):
        return {
            "prog": "neato" if name == "graphviz" else name,
            "args": "-Goverlap=false -Gsplines=true",
        }
    return {}


def _merge_layout_kwargs(
    base: Dict[str, Any],
    user: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]],
    layout: str,
) -> Dict[str, Any]:
    if user is None:
        return base
    lname = layout.lower()
    if isinstance(user, dict) and lname in user and isinstance(user[lname], dict):
        return {**base, **user[lname]}
    return {**base, **user}


# ---------- main: compute positions + normalization (now with kk_enhanced) ----------


def compute_node_positions(
    G: nx.Graph,
    layout: Union[str, Callable] = "circular",
    node_pos: Optional[Dict[Any, Any]] = None,
    seed: Optional[int] = None,
    prog: str = "neato",
    copy_graph: bool = True,
    layout_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
    # normalization controls:
    normalize: Union[bool, str] = "auto",  # True/False/'auto'
    target_half_size: float = 1.0,
    margin: float = 0.0,
    keep_aspect: bool = True,
    flip_y_for_graphviz: bool = False,
    **kwargs,
) -> Dict[Any, Any]:
    """
    Compute node positions for a graph and normalize to [-1,1]^2 when appropriate.

    New layout:
      - 'kk_enhanced' → KK per component + packing + duplicate spread (+ optional spring relax)
                        kw: weight, scale, pad, min_sep, spring_iterations, spring_k, seed
    """
    if node_pos is not None:
        pos = node_pos
    else:
        H = deepcopy(G) if copy_graph else G

        if callable(layout):
            pos = layout(H, **(layout_kwargs or {}), **kwargs)
        else:
            name = str(layout).lower().strip()
            base = _default_kwargs_for_layout(H, name, seed)
            per_layout = _merge_layout_kwargs(base, layout_kwargs, name)
            per_layout.update(kwargs or {})

            if name in ("circular", "circle"):
                pos = nx.circular_layout(H, **per_layout)
            elif name in ("spring", "fr", "fruchterman_reingold"):
                pos = nx.spring_layout(H, **per_layout)
            elif name in ("kamada_kawai", "kk"):
                pos = nx.kamada_kawai_layout(H, **per_layout)
            elif name in ("kk_enhanced",):
                w = per_layout.pop("weight", "weight")
                sc = per_layout.pop("scale", 1.0)
                pdm = per_layout.pop("pad", 2.5)
                msp = per_layout.pop("min_sep", 1e-2)
                sit = int(per_layout.pop("spring_iterations", 0) or 0)
                sk = per_layout.pop("spring_k", None)
                sd = per_layout.get("seed", seed)
                pos = kk_positions_packed(H, weight=w, scale=sc, seed=sd, pad=pdm)
                if msp and msp > 0:
                    pos = spread_duplicates(pos, min_sep=msp, seed=sd)
                if sit > 0:
                    pos = relax_with_spring(H, pos, iterations=sit, k=sk, seed=sd)
            elif name in ("spectral",):
                pos = nx.spectral_layout(H, **per_layout)
            elif name in ("shell",):
                if "nlist" not in per_layout or not per_layout["nlist"]:
                    per_layout = {**per_layout, **{"nlist": _auto_shells(H)}}
                pos = nx.shell_layout(H, **per_layout)
            elif name in ("planar",):
                try:
                    pos = nx.planar_layout(H, **per_layout)
                except Exception:
                    sp_base = _default_kwargs_for_layout(H, "spring", seed)
                    pos = nx.spring_layout(
                        H, **_merge_layout_kwargs(sp_base, layout_kwargs, "spring")
                    )
            elif name in ("random",):
                pos = nx.random_layout(H, **per_layout)
            elif name in ("spiral",):
                per_layout.setdefault("resolution", 0.35)
                per_layout.setdefault("equidistant", False)
                pos = nx.spiral_layout(H, **per_layout)
            elif name in ("graphviz", "neato", "dot", "fdp", "sfdp", "twopi", "circo"):
                prog_to_use = per_layout.pop(
                    "prog", prog if name == "graphviz" else name
                )
                args = per_layout.pop("args", None)
                try:
                    from networkx.drawing.nx_agraph import (
                        graphviz_layout as agraph_layout,
                    )

                    pos = agraph_layout(H, prog=prog_to_use, args=args)
                except Exception:
                    try:
                        from networkx.drawing.nx_pydot import (
                            graphviz_layout as pydot_layout,
                        )

                        pos = pydot_layout(H, prog=prog_to_use, args=args)
                    except Exception:
                        pos = nx.circular_layout(H)
            else:
                pos = nx.circular_layout(H)

    # --- normalization decision ---
    coords = np.array(list(pos.values()), dtype=float)
    if coords.size == 0:
        return pos
    max_abs = np.abs(coords).max()

    is_graphviz_family = False
    if not callable(layout):
        lname = str(layout).lower().strip()
        is_graphviz_family = lname in (
            "graphviz",
            "neato",
            "dot",
            "fdp",
            "sfdp",
            "twopi",
            "circo",
        )

    must_normalize = {
        True: True,
        False: False,
        "auto": bool(is_graphviz_family or (max_abs > target_half_size + 1e-9)),
        "Auto": bool(is_graphviz_family or (max_abs > target_half_size + 1e-9)),
    }.get(normalize, False)

    if must_normalize:
        pos = normalize_positions_to_unit_square(
            pos,
            target_half_size=target_half_size,
            keep_aspect=keep_aspect,
            margin=margin,
            flip_y=bool(flip_y_for_graphviz and is_graphviz_family),
        )

    return pos


def plot_graph(
    graph,
    val_matrix=None,
    var_names=None,
    fig_ax=None,
    figsize=None,
    save_name=None,
    link_colorbar_label="MCI",
    node_colorbar_label="auto-MCI",
    link_width=None,
    link_attribute=None,
    node_pos=None,
    node_layout="neato",
    node_layout_kwargs: Optional[
        Union[Dict[str, Any], Dict[str, Dict[str, Any]]]
    ] = None,
    arrow_linewidth=3.0,
    vmin_edges=-1,
    vmax_edges=1.0,
    edge_ticks=0.4,
    cmap_edges="RdBu_r",
    vmin_nodes=-1,
    vmax_nodes=1.0,
    node_ticks=0.4,
    cmap_nodes="RdBu_r",
    node_size=0.2,
    node_aspect=None,
    # arrowhead_size=20,
    curved_radius=0.2,
    label_fontsize=10,
    tick_label_size=6,
    alpha=1.0,
    node_label_size=12,
    link_label_fontsize=10,
    lag_array=None,
    # network_lower_bound=0.2,
    show_colorbar=True,
    # inner_edge_style="dashed",
    link_matrix=None,
    special_nodes=None,
    edge_color="black",
    target_node=None,
    target_in_color="#2196F3",
    target_out_color="#FF9800",
    target_node_color="#4CAF50",
    target_between_color="#9C27B0",
    show_autodependency_lags=False,
    title=None,
    return_pos=False,
):
    """Creates a network plot.

    The network is defined from links in graph. Nodes
    denote variables, straight links contemporaneous dependencies and curved
    arrows lagged dependencies. The node color denotes the maximal absolute
    auto-dependency and the link color the value at the lag with maximal
    absolute cross-dependency. The link label lists the lags with significant
    dependency in order of absolute magnitude. The network can also be
    plotted over a map drawn before on the same axis. Then the node positions
    can be supplied in appropriate axis coordinates via node_pos.

    Parameters
    ----------
    graph : string or bool array-like, optional (default: None)
        Either string matrix providing graph or bool array providing only adjacencies
        Must be of same shape as val_matrix.
    val_matrix : array_like
        Matrix of shape (N, N, tau_max+1) containing test statistic values.
    var_names : list, optional (default: None)
        List of variable names. If None, range(N) is used.
    fig_ax : tuple of figure and axis object, optional (default: None)
        Figure and axes instance. If None they are created.
    figsize : tuple
        Size of figure.
    save_name : str, optional (default: None)
        Name of figure file to save figure. If None, figure is shown in window.
    link_colorbar_label : str, optional (default: 'MCI')
        Test statistic label.
    node_colorbar_label : str, optional (default: 'auto-MCI')
        Test statistic label for auto-dependencies.
    link_width : array-like, optional (default: None)
        Array of val_matrix.shape specifying relative link width with maximum
        given by arrow_linewidth. If None, all links have same width.
    link_attribute : array-like, optional (default: None)
        String array of val_matrix.shape specifying link attributes.
    node_pos : dictionary, optional (default: None)
        Dictionary of node positions in axis coordinates of form
        node_pos = {'x':array of shape (N,), 'y':array of shape(N)}. These
        coordinates could have been transformed before for basemap plots. You can
        also add a key 'transform':ccrs.PlateCarree() in order to plot graphs on
        a map using cartopy.
    arrow_linewidth : float, optional (default: 30)
        Linewidth.
    vmin_edges : float, optional (default: -1)
        Link colorbar scale lower bound.
    vmax_edges : float, optional (default: 1)
        Link colorbar scale upper bound.
    edge_ticks : float, optional (default: 0.4)
        Link tick mark interval.
    cmap_edges : str, optional (default: 'RdBu_r')
        Colormap for links.
    vmin_nodes : float, optional (default: 0)
        Node colorbar scale lower bound.
    vmax_nodes : float, optional (default: 1)
        Node colorbar scale upper bound.
    node_ticks : float, optional (default: 0.4)
        Node tick mark interval.
    cmap_nodes : str, optional (default: 'OrRd')
        Colormap for links.
    node_size : int, optional (default: 0.3)
        Node size.
    node_aspect : float, optional (default: None)
        Ratio between the heigth and width of the varible nodes.
    arrowhead_size : int, optional (default: 20)
        Size of link arrow head. Passed on to FancyArrowPatch object.
    curved_radius, float, optional (default: 0.2)
        Curvature of links. Passed on to FancyArrowPatch object.
    label_fontsize : int, optional (default: 10)
        Fontsize of colorbar labels.
    alpha : float, optional (default: 1.)
        Opacity.
    node_label_size : int, optional (default: 10)
        Fontsize of node labels.
    link_label_fontsize : int, optional (default: 6)
        Fontsize of link labels.
    tick_label_size : int, optional (default: 6)
        Fontsize of tick labels.
    lag_array : array, optional (default: None)
        Optional specification of lags overwriting np.arange(0, tau_max+1)
    show_colorbar : bool
        Whether to show colorbars for links and nodes.
    edge_color : str, optional (default: 'black')
        Default color for edges when no colormap is active (i.e. when
        ``val_matrix`` is *None* or ``target_node`` is set). Accepts any
        matplotlib color specification.
    target_node : str, int, or list thereof, optional (default: None)
        Variable name(s) or index(es) to highlight. When set, edges to/from
        these nodes are colored with ``target_in_color`` / ``target_out_color``
        and the nodes themselves are colored with ``target_node_color``.
        Overrides colormap-based coloring. Accepts a single value or a list
        for multi-target highlighting. Per-target colors can be specified by
        passing lists of matching length to the color parameters.
    target_in_color : str or list of str, optional (default: '#2196F3')
        Color for edges pointing *into* target node(s).
    target_out_color : str or list of str, optional (default: '#FF9800')
        Color for edges pointing *out of* target node(s).
    target_node_color : str or list of str, optional (default: '#4CAF50')
        Fill color for target node(s).
    target_between_color : str, optional (default: '#9C27B0')
        Color for edges connecting two target nodes.
    show_autodependency_lags : bool (default: False)
        Shows significant autodependencies for a node.
    """

    if link_matrix is not None:
        raise ValueError(
            "link_matrix is deprecated and replaced by graph array"
            " which is now returned by all methods."
        )

    if fig_ax is None:
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on=False)
        if title:
            ax.set_title(title)
    else:
        fig, ax = fig_ax

    graph = np.copy(graph.squeeze())

    if graph.ndim == 4:
        raise ValueError(
            "This graph (N,N,tau_max+1,tau_max+1) cannot be represented by plot_graph."
            "Use plot_time_series_graph instead."
        )

    if graph.ndim == 2:
        # If a non-time series (N,N)-graph is given, insert a dummy dimension
        graph = np.expand_dims(graph, axis=2)

    if val_matrix is None:
        no_coloring = True
        cmap_edges = None
        cmap_nodes = None
    else:
        no_coloring = False

    # Resolve target_node(s) before _check_matrices (needs raw shape)
    target_indices = None
    _target_in_colors = {}
    _target_out_colors = {}
    if target_node is not None:
        raw_N = graph.shape[0]
        if var_names is None:
            _var_names_lookup = list(range(raw_N))
        else:
            _var_names_lookup = list(var_names)

        targets = (
            target_node if isinstance(target_node, (list, tuple)) else [target_node]
        )
        node_colors = (
            target_node_color
            if isinstance(target_node_color, (list, tuple))
            else [target_node_color] * len(targets)
        )
        in_colors = (
            target_in_color
            if isinstance(target_in_color, (list, tuple))
            else [target_in_color] * len(targets)
        )
        out_colors = (
            target_out_color
            if isinstance(target_out_color, (list, tuple))
            else [target_out_color] * len(targets)
        )

        name_to_idx = {n: i for i, n in enumerate(_var_names_lookup)}
        target_indices = set()
        if special_nodes is None:
            special_nodes = {}

        for t, nc, ic, oc in zip(targets, node_colors, in_colors, out_colors):
            if isinstance(t, str):
                if t not in name_to_idx:
                    raise ValueError(f"target_node '{t}' not found in var_names")
                idx = name_to_idx[t]
            else:
                idx = int(t)
            target_indices.add(idx)
            _target_in_colors[idx] = ic
            _target_out_colors[idx] = oc
            target_key = (
                t
                if isinstance(t, str)
                else (_var_names_lookup[idx] if var_names else idx)
            )
            special_nodes[(target_key, 0)] = nc

        no_coloring = True
        cmap_edges = None
        cmap_nodes = None

    graph, val_matrix, link_width, link_attribute = _check_matrices(
        graph, val_matrix, link_width, link_attribute
    )

    N, N, dummy = graph.shape
    tau_max = dummy - 1
    # max_lag = tau_max + 1

    if np.count_nonzero(graph != "") == np.count_nonzero(np.diagonal(graph) != ""):
        diagonal = True
    else:
        diagonal = False

    if np.count_nonzero(graph == "") == graph.size or diagonal:
        graph[0, 1, 0] = "xxx"  # Workaround, will not be plotted...
        no_links = True
    else:
        no_links = False

    if var_names is None:
        var_names = range(N)

    # Define graph links by absolute maximum (positive or negative like for
    # partial correlation)
    # val_matrix[np.abs(val_matrix) < sig_thres] = 0.

    # Only draw link in one direction among contemp
    # Remove lower triangle
    link_matrix_upper = np.copy(graph)
    link_matrix_upper[:, :, 0] = np.triu(link_matrix_upper[:, :, 0])

    # net = _get_absmax(link_matrix != "")
    net = np.any(link_matrix_upper != "", axis=2)
    G = nx.DiGraph(net)

    # This handels Graphs with no links.
    # nx.draw(G, alpha=0, zorder=-10)

    node_color = list(np.zeros(N))

    if show_autodependency_lags:
        autodep_sig_lags = np.full(N, None, dtype="object")
    else:
        autodep_sig_lags = None

    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and lagged links are handled separately
    for u, v, dic in G.edges(data=True):
        dic["no_links"] = no_links
        # average lagfunc for link u --> v ANDOR u -- v
        if tau_max > 0:
            # argmax of absolute maximum where a link exists!
            links = np.where(link_matrix_upper[u, v, 1:] != "")[0]
            if len(links) > 0:
                argmax_links = np.abs(val_matrix[u, v][1:][links]).argmax()
                argmax = links[argmax_links] + 1
            else:
                argmax = 0
        else:
            argmax = 0

        if u != v:
            # For contemp links masking or finite samples can lead to different
            # values for u--v and v--u
            # Here we use the  maximum for the width and weight (=color)
            # of the link
            # Draw link if u--v OR v--u at lag 0 is nonzero
            # dic['inner_edge'] = ((np.abs(val_matrix[u, v][0]) >=
            #                       sig_thres[u, v][0]) or
            #                      (np.abs(val_matrix[v, u][0]) >=
            #                       sig_thres[v, u][0]))
            dic["inner_edge"] = link_matrix_upper[u, v, 0]
            dic["inner_edge_type"] = link_matrix_upper[u, v, 0]
            dic["inner_edge_alpha"] = alpha
            if no_coloring:
                dic["inner_edge_color"] = None
            else:
                dic["inner_edge_color"] = val_matrix[u, v, 0]
            # # value at argmax of average
            # if np.abs(val_matrix[u, v][0] - val_matrix[v, u][0]) > .0001:
            #     print("Contemporaneous I(%d; %d)=%.3f != I(%d; %d)=%.3f" % (
            #           u, v, val_matrix[u, v][0], v, u, val_matrix[v, u][0]) +
            #           " due to conditions, finite sample effects or "
            #           "masking, here edge color = "
            #           "larger (absolute) value.")
            # dic['inner_edge_color'] = _get_absmax(
            #     np.array([[[val_matrix[u, v][0],
            #                    val_matrix[v, u][0]]]])).squeeze()

            if link_width is None:
                dic["inner_edge_width"] = arrow_linewidth
            else:
                dic["inner_edge_width"] = (
                    link_width[u, v, 0] / link_width.max() * arrow_linewidth
                )

            if link_attribute is None:
                dic["inner_edge_attribute"] = None
            else:
                dic["inner_edge_attribute"] = link_attribute[u, v, 0]

            #     # fraction of nonzero values
            dic["inner_edge_style"] = "solid"
            # else:
            # dic['inner_edge_style'] = link_style[
            #         u, v, 0]

            all_strengths.append(dic["inner_edge_color"])

            if tau_max > 0:
                # True if ensemble mean at lags > 0 is nonzero
                # dic['outer_edge'] = np.any(
                #     np.abs(val_matrix[u, v][1:]) >= sig_thres[u, v][1:])
                dic["outer_edge"] = np.any(link_matrix_upper[u, v, 1:] != "")
            else:
                dic["outer_edge"] = False
            # print(u, v, dic["outer_edge"], argmax, link_matrix_upper[u, v, :])

            dic["outer_edge_type"] = link_matrix_upper[u, v, argmax]

            dic["outer_edge_alpha"] = alpha
            if link_width is None:
                # fraction of nonzero values
                dic["outer_edge_width"] = arrow_linewidth
            else:
                dic["outer_edge_width"] = (
                    link_width[u, v, argmax] / link_width.max() * arrow_linewidth
                )

            if link_attribute is None:
                # fraction of nonzero values
                dic["outer_edge_attribute"] = None
            else:
                dic["outer_edge_attribute"] = link_attribute[u, v, argmax]

            # value at argmax of average
            if no_coloring:
                dic["outer_edge_color"] = None
            else:
                dic["outer_edge_color"] = val_matrix[u, v][argmax]
            all_strengths.append(dic["outer_edge_color"])

            # Sorted list of significant lags (only if robust wrt
            # d['min_ensemble_frac'])
            if tau_max > 0:
                lags = np.abs(val_matrix[u, v][1:]).argsort()[::-1] + 1
                sig_lags = (np.where(link_matrix_upper[u, v, 1:] != "")[0] + 1).tolist()
            else:
                lags, sig_lags = [], []
            if lag_array is not None:
                dic["label"] = str(
                    [int(lag_array[la]) for la in lags if la in sig_lags]
                )[1:-1].replace(" ", "")
            else:
                dic["label"] = str([int(la) for la in lags if la in sig_lags])[
                    1:-1
                ].replace(" ", "")
        else:
            # Node color is max of average autodependency
            if no_coloring:
                node_color[u] = None
            else:
                node_color[u] = val_matrix[u, v][argmax]

            if show_autodependency_lags:
                autodep_sig_lags[u] = "\n\n\n" + ",".join(
                    str(i)
                    for i in (
                        np.where(link_matrix_upper[u, v, 1:] != "")[0] + 1
                    ).tolist()
                )
                # Lags upto tau_max
                # autodep_lags = np.argsort(val_matrix[u, v][1:])[::-1]
                # autodep_lags += 1
                # autodeplags[u] = "\n\n\n" + ",".join(str(i) for i in autodep_lags.tolist())

            dic["inner_edge_attribute"] = None
            dic["outer_edge_attribute"] = None

        # dic['outer_edge_edge'] = False
        # dic['outer_edge_edgecolor'] = None
        # dic['inner_edge_edge'] = False
        # dic['inner_edge_edgecolor'] = None

    if target_indices is not None:
        for u, v, dic in G.edges(data=True):
            if u == v:
                continue
            u_is_target = u in target_indices
            v_is_target = v in target_indices
            if u_is_target and v_is_target:
                color = target_between_color
                dic["inner_edge_color"] = color
                dic["outer_edge_color"] = color
            elif v_is_target:
                color = _target_in_colors[v]
                dic["inner_edge_color"] = color
                dic["outer_edge_color"] = color
            elif u_is_target:
                color = _target_out_colors[u]
                dic["inner_edge_color"] = color
                dic["outer_edge_color"] = color

    if special_nodes is not None:
        special_nodes_draw = {}
        name_to_idx = {name: idx for idx, name in enumerate(var_names)}
        for node in special_nodes:
            i, tau = node
            if tau >= -tau_max:
                idx = name_to_idx.get(i, i)
                special_nodes_draw[idx] = special_nodes[node]
        special_nodes = special_nodes_draw

    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.0]

    if node_pos is None:
        # pos = nx.circular_layout(deepcopy(G))
        pos = compute_node_positions(
            G, layout=node_layout, layout_kwargs=node_layout_kwargs, seed=42
        )
    else:
        pos = {}
        for i in range(N):
            pos[i] = (node_pos["x"][i], node_pos["y"][i])

    if node_pos is not None and "transform" in node_pos:
        transform = node_pos["transform"]
    else:
        transform = ax.transData

    if cmap_nodes is None:
        node_color = None

    node_rings = {
        0: {
            "sizes": None,
            "color_array": node_color,
            "cmap": cmap_nodes,
            "vmin": vmin_nodes,
            "vmax": vmax_nodes,
            "ticks": node_ticks,
            "label": node_colorbar_label,
            "colorbar": show_colorbar,
        }
    }

    _draw_network_with_curved_edges(
        fig=fig,
        ax=ax,
        G=deepcopy(G),
        pos=pos,
        node_rings=node_rings,
        # 'vmin':float or None, 'vmax':float or None, 'label':string or None}}
        node_labels=var_names,
        node_label_size=node_label_size,
        # node_alpha=alpha,
        standard_size=node_size,
        node_aspect=node_aspect,
        standard_cmap="OrRd",
        standard_color_nodes="lightgrey",
        standard_color_links=edge_color,
        # log_sizes=False,
        cmap_links=cmap_edges,
        links_vmin=vmin_edges,
        links_vmax=vmax_edges,
        links_ticks=edge_ticks,
        tick_label_size=tick_label_size,
        # arrowstyle="simple",
        # arrowhead_size=arrowhead_size,
        curved_radius=curved_radius,
        label_fontsize=label_fontsize,
        link_label_fontsize=link_label_fontsize,
        link_colorbar_label=link_colorbar_label,
        # network_lower_bound=network_lower_bound,
        show_colorbar=show_colorbar,
        # label_fraction=label_fraction,
        special_nodes=special_nodes,
        autodep_sig_lags=autodep_sig_lags,
        show_autodependency_lags=show_autodependency_lags,
        transform=transform,
    )

    # Convert internal pos dict to node_pos format for reuse
    node_pos_out = {
        "x": np.array([pos[i][0] for i in range(N)]),
        "y": np.array([pos[i][1] for i in range(N)]),
    }

    if save_name is not None:
        pyplot.savefig(save_name, dpi=300)
    else:
        if return_pos:
            return fig, ax, node_pos_out
        return fig, ax


def _reverse_patt(patt):
    """Inverts a link pattern"""

    if patt == "":
        return ""

    left_mark, middle_mark, right_mark = patt[0], patt[1], patt[2]
    if left_mark == "<":
        new_right_mark = ">"
    else:
        new_right_mark = left_mark
    if right_mark == ">":
        new_left_mark = "<"
    else:
        new_left_mark = right_mark

    return new_left_mark + middle_mark + new_right_mark

    # if patt in ['---', 'o--', '--o', 'o-o', '']:
    #     return patt[::-1]
    # elif patt == '<->':
    #     return '<->'
    # elif patt == 'o->':
    #     return '<-o'
    # elif patt == '<-o':
    #     return 'o->'
    # elif patt == '-->':
    #     return '<--'
    # elif patt == '<--':
    #     return '-->'


def _check_matrices(graph, val_matrix, link_width, link_attribute):
    if graph.dtype != "<U3":
        # Transform to new graph data type U3
        old_matrix = np.copy(graph)
        graph = np.zeros(old_matrix.shape, dtype="<U3")
        graph[:] = ""
        for i, j, tau in zip(*np.where(old_matrix)):
            if tau == 0:
                if old_matrix[j, i, 0] == 0:
                    graph[i, j, 0] = "-->"
                    graph[j, i, 0] = "<--"
                else:
                    graph[i, j, 0] = "o-o"
                    graph[j, i, 0] = "o-o"
            else:
                graph[i, j, tau] = "-->"
    if graph.ndim == 4:
        for i, j, taui, tauj in zip(*np.where(graph)):
            if graph[i, j, taui, tauj] not in [
                "---",
                "o--",
                "--o",
                "o-o",
                "o->",
                "<-o",
                "-->",
                "<--",
                "<->",
                "x-o",
                "o-x",
                "x--",
                "--x",
                "x->",
                "<-x",
                "x-x",
                "<-+",
                "+->",
            ]:
                raise ValueError("Invalid graph entry.")
            if graph[i, j, taui, tauj] != _reverse_patt(graph[j, i, tauj, taui]):
                raise ValueError(
                    "graph needs to have consistent entries: "
                    "graph[i, j, taui, tauj] == _reverse_patt(graph[j, i, tauj, taui])"
                )
            if (
                val_matrix is not None
                and val_matrix[i, j, taui, tauj] != val_matrix[j, i, tauj, taui]
            ):
                raise ValueError(
                    "val_matrix needs to have consistent entries: "
                    "val_matrix[i, j, taui, tauj] == val_matrix[j, i, tauj, taui]"
                )
            if (
                link_width is not None
                and link_width[i, j, taui, tauj] != link_width[j, i, tauj, taui]
            ):
                raise ValueError(
                    "link_width needs to have consistent entries: "
                    "link_width[i, j, taui, tauj] == link_width[j, i, tauj, taui]"
                )
            if (
                link_attribute is not None
                and link_attribute[i, j, taui, tauj] != link_attribute[j, i, tauj, taui]
            ):
                raise ValueError(
                    "link_attribute needs to have consistent entries: "
                    "link_attribute[i, j, taui, tauj] == link_attribute[j, i, tauj, taui]"
                )
    else:
        # print(graph[:,:,0])
        # Assert that graph has valid and consistent lag-zero entries
        for i, j, tau in zip(*np.where(graph)):
            if tau == 0:
                if graph[i, j, 0] != _reverse_patt(graph[j, i, 0]):
                    raise ValueError(
                        f"graph needs to have consistent lag-zero links, but "
                        f"graph[{i},{j},0]={graph[i, j, 0]} and graph[{j},{i},0]={graph[j, i, 0]}"
                    )
                if (
                    val_matrix is not None
                    and val_matrix[i, j, 0] != val_matrix[j, i, 0]
                ):
                    raise ValueError("val_matrix needs to be symmetric for lag-zero")
                if (
                    link_width is not None
                    and link_width[i, j, 0] != link_width[j, i, 0]
                ):
                    raise ValueError("link_width needs to be symmetric for lag-zero")
                if (
                    link_attribute is not None
                    and link_attribute[i, j, 0] != link_attribute[j, i, 0]
                ):
                    raise ValueError(
                        "link_attribute needs to be symmetric for lag-zero"
                    )

            if graph[i, j, tau] not in [
                "---",
                "o--",
                "--o",
                "o-o",
                "o->",
                "<-o",
                "-->",
                "<--",
                "<->",
                "x-o",
                "o-x",
                "x--",
                "--x",
                "x->",
                "<-x",
                "x-x",
                "<-+",
                "+->",
            ]:
                raise ValueError("Invalid graph entry.")

    if val_matrix is None:
        # if graph.ndim == 4:
        #     val_matrix = (graph != "").astype("int")
        # else:
        val_matrix = (graph != "").astype("int")

    if link_width is not None and not np.all(link_width >= 0.0):
        raise ValueError("link_width must be non-negative")

    return graph, val_matrix, link_width, link_attribute


def _barycenter_order(graph, max_iter=10):
    """Compute variable ordering that minimizes edge crossings (barycenter heuristic)."""
    graph_bool = graph.astype(bool) if graph.dtype != bool else graph
    N = graph_bool.shape[0]

    # Build undirected neighbor adjacency (i connects to j if any lag has an edge)
    adj = np.zeros((N, N), dtype=bool)
    for tau in range(graph_bool.shape[2]):
        layer = graph_bool[:, :, tau]
        adj |= layer
        if tau > 0:
            adj |= layer.T
    np.fill_diagonal(adj, False)

    order = list(range(N))
    for _ in range(max_iter):
        pos = {v: i for i, v in enumerate(order)}
        bary = []
        for v in range(N):
            neighbors = np.where(adj[v])[0]
            if len(neighbors) > 0:
                bary.append((v, np.mean([pos[n] for n in neighbors])))
            else:
                bary.append((v, pos[v]))
        new_order = [v for v, _ in sorted(bary, key=lambda x: x[1])]
        if new_order == order:
            break
        order = new_order
    return order


def plot_time_series_graph(
    graph,
    val_matrix=None,
    var_names=None,
    fig_ax=None,
    figsize=None,
    link_colorbar_label="MCI",
    save_name=None,
    link_width=None,
    link_attribute=None,
    arrow_linewidth=4,
    vmin_edges=-1,
    vmax_edges=1.0,
    edge_ticks=0.4,
    cmap_edges="RdBu_r",
    order="auto",
    node_size=0.1,
    node_aspect=None,
    # arrowhead_size=20,
    curved_radius=0.2,
    label_fontsize=10,
    node_label_size=10,
    link_label_fontsize=6,
    tick_label_size=6,
    alpha=1.0,
    inner_edge_style="dashed",
    link_matrix=None,
    special_nodes=None,
    node_classification=None,
    # aux_graph=None,
    edge_color="black",
    standard_color_nodes="lightgrey",
    lag_array=None,
    title=None,
    title_pad=None,
):
    """Creates a time series graph.

    The time series graph's links are colored by
    val_matrix.

    Parameters
    ----------
    graph : string or bool array-like, optional (default: None)
        Either string matrix providing graph or bool array providing only adjacencies
        Either of shape (N, N, tau_max + 1) or as auxiliary graph of dims
        (N, N, tau_max+1, tau_max+1) describing auxADMG.
    val_matrix : array_like
        Matrix of same shape as graph containing test statistic values.
    var_names : list, optional (default: None)
        List of variable names. If None, range(N) is used.
    fig_ax : tuple of figure and axis object, optional (default: None)
        Figure and axes instance. If None they are created.
    figsize : tuple
        Size of figure.
    save_name : str, optional (default: None)
        Name of figure file to save figure. If None, figure is shown in window.
    link_colorbar_label : str, optional (default: 'MCI')
        Test statistic label.
    link_width : array-like, optional (default: None)
        Array of val_matrix.shape specifying relative link width with maximum
        given by arrow_linewidth. If None, all links have same width.
    link_attribute : array-like, optional (default: None)
        Array of graph.shape specifying specific in drawing the graph
    order : list or str, optional (default: 'auto')
        Order of variables from top to bottom. ``'auto'`` uses the barycenter
        heuristic to minimize edge crossings. Pass a list of indices for
        manual ordering.
    arrow_linewidth : float, optional (default: 4)
        Linewidth.
    vmin_edges : float, optional (default: -1)
        Link colorbar scale lower bound.
    vmax_edges : float, optional (default: 1)
        Link colorbar scale upper bound.
    edge_ticks : float, optional (default: 0.4)
        Link tick mark interval.
    cmap_edges : str, optional (default: 'RdBu_r')
        Colormap for links.
    node_size : int, optional (default: 0.1)
        Node size.
    node_aspect : float, optional (default: None)
        Ratio between the heigth and width of the varible nodes.
    lag_array : array-like, optional (default: None)
        Array of actual lag values for column labels. If None, uses
        sequential integers 0, 1, ..., tau_max.
    arrowhead_size : int, optional (default: 20)
        Size of link arrow head. Passed on to FancyArrowPatch object.
    curved_radius, float, optional (default: 0.2)
        Curvature of links. Passed on to FancyArrowPatch object.
    label_fontsize : int, optional (default: 10)
        Fontsize of colorbar labels.
    alpha : float, optional (default: 1.)
        Opacity.
    node_label_size : int, optional (default: 10)
        Fontsize of node labels.
    node_label_size : int, optional (default: 10)
        Fontsize of node (variable) labels.
    link_label_fontsize : int, optional (default: 6)
        Fontsize of lag labels on edges.
    tick_label_size : int, optional (default: 6)
        Fontsize of tick labels.
    edge_color : str, optional (default: 'black')
        Color for edges/links when no colormap is active (i.e., when
        ``val_matrix`` is *None*). Also sets the node ring colour.
    title : str, optional (default: None)
        Axes title.
    title_pad : float, optional (default: None)
        Distance in points between the lag-label row and the title.
        Defaults to ``label_fontsize * 2``, which is enough to clear the
        lag labels (``t``, ``t-1``, …) that are drawn just above the axes
        top. Increase if the title still overlaps on very small figures.
    inner_edge_style : string, optional (default: 'dashed')
        Style of inner_edge contemporaneous links.
    special_nodes : dict
        Dictionary of format {(i, -tau): 'blue', ...} to color special nodes.
    node_classification : dict or None (default: None)
        Dictionary of format ``{i: 'space_context', ...}`` to classify
        nodes into system, context, or dummy nodes. Keys are from
        ``{0, ..., N-1}``. Values are ``"system"``, ``"time_context"``,
        ``"space_context"``, ``"time_dummy"``, or ``"space_dummy"``.
        Space contexts and dummy nodes are shown as a single node.
        If None, all nodes are treated as system nodes.
    """

    if link_matrix is not None:
        raise ValueError(
            "link_matrix is deprecated and replaced by graph array"
            " which is now returned by all methods."
        )

    if fig_ax is None:
        fig = pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on=False)
    else:
        fig, ax = fig_ax

    if val_matrix is None:
        no_coloring = True
        cmap_edges = None
    else:
        no_coloring = False

    # Save original graph before string conversion for barycenter ordering
    graph_orig = np.copy(np.asarray(graph).squeeze())

    graph, val_matrix, link_width, link_attribute = _check_matrices(
        graph, val_matrix, link_width, link_attribute
    )

    if graph.ndim == 4:
        N, N, dummy, _ = graph.shape
        tau_max = dummy - 1
        max_lag = tau_max + 1
    else:
        N, N, dummy = graph.shape
        tau_max = dummy - 1
        max_lag = tau_max + 1

    if np.count_nonzero(graph == "") == graph.size:
        if graph.ndim == 4:
            graph[0, 1, 0, 0] = "---"
        else:
            graph[0, 1, 0] = "---"
        no_links = True
    else:
        no_links = False

    if var_names is None:
        var_names = range(N)

    if order is None or order == "auto":
        order = _barycenter_order(graph_orig)

    if set(order) != set(range(N)):
        raise ValueError("order must be a permutation of range(N)")

    def translate(row, lag):
        return row * max_lag + lag

    # Define graph links by absolute maximum (positive or negative like for
    # partial correlation)
    tsg = np.zeros((N * max_lag, N * max_lag))
    tsg_val = np.zeros((N * max_lag, N * max_lag))
    tsg_width = np.zeros((N * max_lag, N * max_lag))
    tsg_style = np.zeros((N * max_lag, N * max_lag), dtype=graph.dtype)
    if link_attribute is not None:
        tsg_attr = np.zeros((N * max_lag, N * max_lag), dtype=link_attribute.dtype)

    if graph.ndim == 4:
        # 4-dimensional graphs represent the finite-time window projection of stationary 3-d graphs
        # They are internally created in some classes
        # Only draw link in one direction
        for i, j, taui, tauj in np.column_stack(np.where(graph)):
            tau = taui - tauj
            # if tau <= 0 and j <= i:
            if translate(i, max_lag - 1 - taui) >= translate(j, max_lag - 1 - tauj):
                continue
            # print(max_lag, (i, -taui), (j, -tauj), aux_graph[i, j, taui, tauj])
            # print(translate(i, max_lag - 1 - taui), translate(j, max_lag-1-tauj))
            tsg[translate(i, max_lag - 1 - taui), translate(j, max_lag - 1 - tauj)] = (
                1.0
            )
            tsg_val[
                translate(i, max_lag - 1 - taui), translate(j, max_lag - 1 - tauj)
            ] = val_matrix[i, j, taui, tauj]
            tsg_style[
                translate(i, max_lag - 1 - taui), translate(j, max_lag - 1 - tauj)
            ] = graph[i, j, taui, tauj]
            if link_width is not None:
                tsg_width[
                    translate(i, max_lag - 1 - taui), translate(j, max_lag - 1 - tauj)
                ] = (link_width[i, j, taui, tauj] / link_width.max() * arrow_linewidth)
            if link_attribute is not None:
                tsg_attr[
                    translate(i, max_lag - 1 - taui), translate(j, max_lag - 1 - tauj)
                ] = link_attribute[
                    i, j, taui, tauj
                ]  # 'spurious'
        # print(tsg_style)

    else:
        # Only draw link in one direction
        # Remove lower triangle
        link_matrix_tsg = np.copy(graph)
        link_matrix_tsg[:, :, 0] = np.triu(graph[:, :, 0])

        for i, j, tau in np.column_stack(np.where(link_matrix_tsg)):
            for t in range(max_lag):
                if (
                    0 <= translate(i, t - tau)
                    and translate(i, t - tau) % max_lag <= translate(j, t) % max_lag
                ):
                    tsg[translate(i, t - tau), translate(j, t)] = (
                        1.0  # val_matrix[i, j, tau]
                    )
                    tsg_val[translate(i, t - tau), translate(j, t)] = val_matrix[
                        i, j, tau
                    ]
                    tsg_style[translate(i, t - tau), translate(j, t)] = graph[i, j, tau]
                    if link_width is not None:
                        tsg_width[translate(i, t - tau), translate(j, t)] = (
                            link_width[i, j, tau] / link_width.max() * arrow_linewidth
                        )
                    if link_attribute is not None:
                        tsg_attr[translate(i, t - tau), translate(j, t)] = (
                            link_attribute[i, j, tau]
                        )

    G = nx.DiGraph(tsg)

    if special_nodes is not None:
        special_nodes_tsg = {}
        for node in special_nodes:
            i, tau = node
            if tau >= -tau_max:
                special_nodes_tsg[translate(i, max_lag - 1 + tau)] = special_nodes[node]

        special_nodes = special_nodes_tsg

    if node_classification is None:
        node_classification = {i: "system" for i in range(N)}
    node_classification_tsg = {}
    for node in node_classification:
        for tau in range(max_lag):
            if tau == 0:
                suffix = "_first"
            elif tau == max_lag - 1:
                suffix = "_last"
            else:
                suffix = "_middle"
            node_classification_tsg[translate(node, tau)] = (
                node_classification[node] + suffix
            )

    # node_color = np.zeros(N)
    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and lagged links are handled separately
    for u, v, dic in G.edges(data=True):
        dic["no_links"] = no_links
        if u != v:
            # tau = np.abs((u - v) % max_lag)
            # Determine neighbors in TSG
            i = u // max_lag
            taui = -(max_lag - 1 - (u % max_lag))
            j = v // max_lag
            tauj = -(max_lag - 1 - (v % max_lag))

            if np.abs(i - j) <= 1 and np.abs(tauj - taui) <= 1:
                inout = "inner"
                dic["inner_edge"] = True
                dic["outer_edge"] = False
            else:
                inout = "outer"
                dic["inner_edge"] = False
                dic["outer_edge"] = True

            dic[f"{inout}_edge_type"] = tsg_style[u, v]

            dic[f"{inout}_edge_alpha"] = alpha

            if link_width is None:
                # fraction of nonzero values
                dic[f"{inout}_edge_width"] = arrow_linewidth
            else:
                dic[f"{inout}_edge_width"] = tsg_width[u, v]

            if link_attribute is None:
                dic[f"{inout}_edge_attribute"] = None
            else:
                dic[f"{inout}_edge_attribute"] = tsg_attr[u, v]

            # value at argmax of average
            if no_coloring:
                dic[f"{inout}_edge_color"] = None
            else:
                dic[f"{inout}_edge_color"] = tsg_val[u, v]

            all_strengths.append(dic[f"{inout}_edge_color"])
            dic["label"] = None

    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.0]

    posarray = np.zeros((N * max_lag, 2))
    for i in range(N * max_lag):
        posarray[i] = np.array([(i % max_lag), (1.0 - i // max_lag)])

    pos_tmp = {}
    for i in range(N * max_lag):
        # for n in range(N):
        #     for tau in range(max_lag):
        #         i = n*N + tau
        pos_tmp[i] = np.array(
            [
                ((i % max_lag) - posarray.min(axis=0)[0])
                / (posarray.max(axis=0)[0] - posarray.min(axis=0)[0]),
                ((1.0 - i // max_lag) - posarray.min(axis=0)[1])
                / (posarray.max(axis=0)[1] - posarray.min(axis=0)[1]),
            ]
        )
        pos_tmp[i][np.isnan(pos_tmp[i])] = 0.0

    pos = {}
    for n in range(N):
        for tau in range(max_lag):
            pos[n * max_lag + tau] = pos_tmp[order[n] * max_lag + tau]

    node_rings = {
        0: {
            "sizes": None,
            "color_array": None,
            "label": "",
            "colorbar": False,
        }
    }

    node_labels = ["" for i in range(N * max_lag)]

    if graph.ndim == 4 and val_matrix is None:
        show_colorbar = False
    else:
        show_colorbar = True

    _draw_network_with_curved_edges(
        fig=fig,
        ax=ax,
        G=deepcopy(G),
        pos=pos,
        node_rings=node_rings,
        node_labels=node_labels,
        node_label_size=node_label_size,
        standard_size=node_size,
        node_aspect=node_aspect,
        standard_cmap="OrRd",
        standard_color_nodes=standard_color_nodes,
        standard_color_links=edge_color,
        cmap_links=cmap_edges,
        links_vmin=vmin_edges,
        links_vmax=vmax_edges,
        links_ticks=edge_ticks,
        link_label_fontsize=link_label_fontsize,
        curved_radius=curved_radius,
        label_fontsize=label_fontsize,
        tick_label_size=tick_label_size,
        link_colorbar_label=link_colorbar_label,
        inner_edge_curved=False,
        special_nodes=special_nodes,
        show_colorbar=show_colorbar,
        node_classification=node_classification_tsg,
        max_lag=max_lag,
    )

    for i in range(N):
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            0.0,
            pos[order[i] * max_lag][1],
            f"{var_names[order[i]]}",
            fontsize=node_label_size,
            horizontalalignment="right",
            verticalalignment="center",
            transform=trans,
        )

    for tau in np.arange(max_lag - 1, -1, -1):
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        lag_val = max_lag - tau - 1
        if tau == max_lag - 1:
            label = r"$t$"
        elif lag_array is not None:
            label = rf"$t-{lag_array[lag_val]}$"
        else:
            label = rf"$t-{lag_val}$"
        ax.text(
            pos[tau][0],
            1.0,
            label,
            fontsize=int(label_fontsize * 0.8),
            horizontalalignment="center",
            verticalalignment="bottom",
            transform=trans,
        )

    if title is not None:
        pad = title_pad if title_pad is not None else label_fontsize * 2
        ax.set_title(title, fontsize=label_fontsize, pad=pad)

    if save_name is not None:
        pyplot.savefig(save_name, dpi=300)
    else:
        return fig, ax


# ---------- lag function matrix plots ----------


def _make_nice_axes(ax, where=None, skip=1, color=None):
    """Makes nice axes."""
    if where is None:
        where = ["left", "bottom"]
    if color is None:
        color = {"left": "black", "right": "black", "bottom": "black", "top": "black"}

    if isinstance(skip, int):
        skip_x = skip_y = skip
    else:
        skip_x = skip[0]
        skip_y = skip[1]

    for loc, spine in ax.spines.items():
        if loc in where:
            spine.set_position(("outward", 5))
            spine.set_color(color[loc])
            if loc in ("left", "right"):
                pyplot.setp(ax.get_yticklines(), color=color[loc])
                pyplot.setp(ax.get_yticklabels(), color=color[loc])
            if loc in ("top", "bottom"):
                pyplot.setp(ax.get_xticklines(), color=color[loc])
        elif loc in [
            item for item in ["left", "bottom", "right", "top"] if item not in where
        ]:
            spine.set_color("none")
        else:
            raise ValueError(f"unknown spine location: {loc}")

    if "top" in where and "bottom" not in where:
        ax.xaxis.set_ticks_position("top")
        if skip_x > 1:
            ax.set_xticks(ax.get_xticks()[::skip_x])
    elif "bottom" in where:
        ax.xaxis.set_ticks_position("bottom")
        if skip_x > 1:
            ax.set_xticks(ax.get_xticks()[::skip_x])
    else:
        ax.xaxis.set_ticks_position("none")
        ax.xaxis.set_ticklabels([])
    if "right" in where and "left" not in where:
        ax.yaxis.set_ticks_position("right")
        if skip_y > 1:
            ax.set_yticks(ax.get_yticks()[::skip_y])
    elif "left" in where:
        ax.yaxis.set_ticks_position("left")
        if skip_y > 1:
            ax.set_yticks(ax.get_yticks()[::skip_y])
    else:
        ax.yaxis.set_ticks_position("none")
        ax.yaxis.set_ticklabels([])

    ax.patch.set_alpha(0.0)


class setup_matrix:
    """Create matrix of lag function panels.

    Parameters
    ----------
    N : int
        Number of variables.
    tau_max : int
        Maximum time lag.
    var_names : list, optional
        Variable names.
    figsize : tuple, optional
        Figure size.
    minimum : float, optional (default: -1.)
        Lower y-axis limit.
    maximum : float, optional (default: 1.)
        Upper y-axis limit.
    label_space_left : float, optional (default: 0.1)
        Fraction of horizontal figure space for left labels.
    label_space_top : float, optional (default: 0.05)
        Fraction of vertical figure space for top labels.
    legend_width : float, optional (default: 0.15)
        Fraction of horizontal figure space for legend.
    x_base : float, optional (default: 1.)
        x-tick interval.
    y_base : float, optional (default: .4)
        y-tick interval.
    plot_gridlines : bool, optional (default: False)
        Whether to show a grid.
    lag_units : str, optional (default: '')
        Units for lag axis label.
    lag_array : array, optional
        Custom lag labels.
    label_fontsize : int, optional (default: 10)
        Fontsize for variable labels.
    """

    def __init__(
        self,
        N,
        tau_max,
        var_names=None,
        figsize=None,
        minimum=-1,
        maximum=1,
        label_space_left=0.1,
        label_space_top=0.05,
        legend_width=0.15,
        legend_fontsize=10,
        x_base=1.0,
        y_base=0.5,
        tick_label_size=6,
        plot_gridlines=False,
        lag_units="",
        lag_array=None,
        label_fontsize=10,
    ):
        self.tau_max = tau_max
        self.labels = []
        self.lag_units = lag_units
        self.lag_array = lag_array
        self.x_base = 1 if x_base is None else x_base
        self.legend_width = legend_width
        self.legend_fontsize = legend_fontsize
        self.label_space_left = label_space_left
        self.label_space_top = label_space_top
        self.label_fontsize = label_fontsize

        self.fig = pyplot.figure(figsize=figsize)
        self.axes_dict = {}

        if var_names is None:
            var_names = range(N)

        plot_index = 1
        for i in range(N):
            for j in range(N):
                self.axes_dict[(i, j)] = self.fig.add_subplot(N, N, plot_index)
                if j == 0:
                    trans = transforms.blended_transform_factory(
                        self.fig.transFigure, self.axes_dict[(i, j)].transAxes
                    )
                    self.axes_dict[(i, j)].text(
                        0.01,
                        0.5,
                        f"{var_names[i]}",
                        fontsize=label_fontsize,
                        horizontalalignment="left",
                        verticalalignment="center",
                        transform=trans,
                    )
                if i == 0:
                    trans = transforms.blended_transform_factory(
                        self.axes_dict[(i, j)].transAxes, self.fig.transFigure
                    )
                    self.axes_dict[(i, j)].text(
                        0.5,
                        0.99,
                        rf"${{\to}}$ {var_names[j]}",
                        fontsize=label_fontsize,
                        horizontalalignment="center",
                        verticalalignment="top",
                        transform=trans,
                    )

                _make_nice_axes(
                    self.axes_dict[(i, j)], where=["left", "bottom"], skip=(1, 1)
                )
                if x_base is not None:
                    self.axes_dict[(i, j)].xaxis.set_major_locator(
                        ticker.FixedLocator(np.arange(0, self.tau_max + 1, x_base))
                    )
                    if x_base / 2.0 % 1 == 0:
                        self.axes_dict[(i, j)].xaxis.set_minor_locator(
                            ticker.FixedLocator(
                                np.arange(0, self.tau_max + 1, x_base / 2.0)
                            )
                        )
                if y_base is not None:
                    self.axes_dict[(i, j)].yaxis.set_major_locator(
                        ticker.FixedLocator(
                            np.arange(
                                _myround(minimum, y_base, "down"),
                                _myround(maximum, y_base, "up") + y_base,
                                y_base,
                            )
                        )
                    )
                    self.axes_dict[(i, j)].yaxis.set_minor_locator(
                        ticker.FixedLocator(
                            np.arange(
                                _myround(minimum, y_base, "down"),
                                _myround(maximum, y_base, "up") + y_base,
                                y_base / 2.0,
                            )
                        )
                    )
                    self.axes_dict[(i, j)].set_ylim(
                        _myround(minimum, y_base, "down"),
                        _myround(maximum, y_base, "up"),
                    )
                if j != 0:
                    self.axes_dict[(i, j)].get_yaxis().set_ticklabels([])
                self.axes_dict[(i, j)].set_xlim(0, self.tau_max)
                if plot_gridlines:
                    self.axes_dict[(i, j)].grid(
                        True,
                        which="major",
                        color="black",
                        linestyle="dotted",
                        dashes=(1, 1),
                        linewidth=0.05,
                        zorder=-5,
                    )
                self.axes_dict[(i, j)].tick_params(
                    axis="both", which="major", labelsize=tick_label_size
                )
                self.axes_dict[(i, j)].tick_params(
                    axis="both", which="minor", labelsize=tick_label_size
                )
                plot_index += 1

    def add_lagfuncs(
        self,
        val_matrix,
        sig_thres=None,
        conf_matrix=None,
        color="black",
        label=None,
        two_sided_thres=True,
        marker=".",
        markersize=5,
        alpha=1.0,
    ):
        """Add lag function plot from val_matrix array.

        Parameters
        ----------
        val_matrix : array_like
            Matrix of shape (N, N, tau_max+1) containing test statistic values.
        sig_thres : array-like, optional
            Significance thresholds of same shape as val_matrix.
        conf_matrix : array-like, optional
            Confidence bounds of shape (N, N, tau_max+1, 2).
        color : str, optional (default: 'black')
            Line color.
        label : str, optional
            Test statistic label.
        two_sided_thres : bool, optional (default: True)
            Whether to draw sig_thres for pos. and neg. values.
        marker : str, optional (default: '.')
            Marker symbol.
        markersize : int, optional (default: 5)
            Marker size.
        alpha : float, optional (default: 1.)
            Opacity.
        """
        if label is not None:
            self.labels.append((label, color, marker, markersize, alpha))

        for ij in list(self.axes_dict):
            i = ij[0]
            j = ij[1]
            maskedres = np.copy(val_matrix[i, j, 0:])
            self.axes_dict[(i, j)].plot(
                range(0, self.tau_max + 1),
                maskedres,
                linestyle="",
                color=color,
                marker=marker,
                markersize=markersize,
                alpha=alpha,
                clip_on=False,
            )
            if conf_matrix is not None:
                maskedconfres = np.copy(conf_matrix[i, j, 0:])
                self.axes_dict[(i, j)].plot(
                    range(0, self.tau_max + 1),
                    maskedconfres[:, 0],
                    linestyle="",
                    color=color,
                    marker="_",
                    markersize=markersize - 2,
                    alpha=alpha,
                    clip_on=False,
                )
                self.axes_dict[(i, j)].plot(
                    range(0, self.tau_max + 1),
                    maskedconfres[:, 1],
                    linestyle="",
                    color=color,
                    marker="_",
                    markersize=markersize - 2,
                    alpha=alpha,
                    clip_on=False,
                )

            self.axes_dict[(i, j)].plot(
                range(0, self.tau_max + 1),
                np.zeros(self.tau_max + 1 - 0),
                color="black",
                linestyle="dotted",
                linewidth=0.1,
            )

            if sig_thres is not None:
                maskedsigres = sig_thres[i, j, 0:]
                self.axes_dict[(i, j)].plot(
                    range(0, self.tau_max + 1),
                    maskedsigres,
                    color=color,
                    linestyle="solid",
                    linewidth=0.1,
                    alpha=alpha,
                )
                if two_sided_thres:
                    self.axes_dict[(i, j)].plot(
                        range(0, self.tau_max + 1),
                        -sig_thres[i, j, 0:],
                        color=color,
                        linestyle="solid",
                        linewidth=0.1,
                        alpha=alpha,
                    )

    def savefig(self, name=None):
        """Save matrix figure.

        Parameters
        ----------
        name : str, optional
            File name. If None, figure is shown in window.
        """
        if len(self.labels) > 0:
            axlegend = self.fig.add_subplot(111, frameon=False)
            axlegend.spines["left"].set_color("none")
            axlegend.spines["right"].set_color("none")
            axlegend.spines["bottom"].set_color("none")
            axlegend.spines["top"].set_color("none")
            axlegend.set_xticks([])
            axlegend.set_yticks([])

            for item in self.labels:
                label, color, marker_, markersize_, alpha_ = item
                axlegend.plot(
                    [],
                    [],
                    linestyle="",
                    color=color,
                    marker=marker_,
                    markersize=markersize_,
                    label=label,
                    alpha=alpha_,
                )
            axlegend.legend(
                loc="upper left",
                ncol=1,
                bbox_to_anchor=(1.05, 0.0, 0.1, 1.0),
                borderaxespad=0,
                fontsize=self.legend_fontsize,
            ).draw_frame(False)

            self.fig.subplots_adjust(
                left=self.label_space_left,
                right=1.0 - self.legend_width,
                top=1.0 - self.label_space_top,
                hspace=0.35,
                wspace=0.35,
            )
            pyplot.figtext(
                0.5,
                0.01,
                f"lag $\\tau$ [{self.lag_units}]",
                horizontalalignment="center",
                fontsize=self.label_fontsize,
            )
        else:
            self.fig.subplots_adjust(
                left=self.label_space_left,
                right=0.95,
                top=1.0 - self.label_space_top,
                hspace=0.35,
                wspace=0.35,
            )
            pyplot.figtext(
                0.55,
                0.01,
                f"lag $\\tau$ [{self.lag_units}]",
                horizontalalignment="center",
                fontsize=self.label_fontsize,
            )

        if self.lag_array is not None:
            assert self.lag_array.shape == np.arange(self.tau_max + 1).shape
            for ij in list(self.axes_dict):
                i = ij[0]
                j = ij[1]
                self.axes_dict[(i, j)].set_xticklabels(self.lag_array[:: self.x_base])

        if name is not None:
            self.fig.savefig(name)
        else:
            return self.fig, self.axes_dict


def plot_lagfuncs(
    val_matrix, name=None, setup_args=None, add_lagfunc_args=None, dpi=300
):
    """Wrapper to plot lag functions.

    Parameters
    ----------
    val_matrix : array_like
        Matrix of shape (N, N, tau_max+1) containing test statistic values.
    name : str, optional
        File name. If None, figure is shown in window.
    setup_args : dict, optional
        Arguments for setup_matrix.
    add_lagfunc_args : dict, optional
        Arguments for add_lagfuncs.

    Returns
    -------
    matrix : setup_matrix
        The matrix object (has .fig attribute).
    """
    if setup_args is None:
        setup_args = {}
    if add_lagfunc_args is None:
        add_lagfunc_args = {}

    N, N, tau_max_plusone = val_matrix.shape
    tau_max = tau_max_plusone - 1

    matrix = setup_matrix(N=N, tau_max=tau_max, **setup_args)
    matrix.add_lagfuncs(val_matrix=val_matrix, **add_lagfunc_args)
    try:
        matrix.savefig(name=name, dpi=dpi)
    except TypeError:
        matrix.savefig(name=name)

    return matrix
