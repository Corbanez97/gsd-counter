#!/usr/bin/env python3
"""Regenerate every figure in README.md into figures/.  Takes about half a minute.

    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py models     # only the ones whose name contains "models"
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gsd import (HAAH_CUBIC, NEWMAN_MOORE, STENCILS, TORIC_CODE, Poly, classical,  # noqa: E402
                 fractals, rank, ring_pow)

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")

# poster palette: navy ink, X = blue, Z = orange, magnitudes in one teal hue
INK, MUTED, GRID = "#1f2a44", "#6b7280", "#e5e7eb"
BLUE, ORANGE, TEAL = "#2f6fd0", "#e0701a", "#17917a"
TEALS = ListedColormap(["#ffffff", TEAL])

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def bare(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def lattice(ax, n=5):
    for i in range(n):
        ax.plot([0, n - 1], [i, i], color=GRID, lw=1, zorder=0)
        ax.plot([i, i], [0, n - 1], color=GRID, lw=1, zorder=0)
    ax.scatter(*np.meshgrid(range(n), range(n)), s=6, color=MUTED, zorder=1)
    ax.set_aspect("equal")
    bare(ax)


def spin(ax, x, y, label="Z", color=ORANGE, size=14):
    ax.add_patch(plt.Circle((x, y), 0.26, color=color, zorder=3))
    ax.text(x, y, label, color="white", ha="center", va="center", fontsize=size * 0.55,
            fontweight="bold", zorder=4)


# =================================================================== 1. the pipeline
def fig_pipeline():
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    ax = axes[0]
    lattice(ax)
    for (i, j), lab in {(0, 0): "1", (1, 0): "x", (2, 0): "x²", (0, 1): "y", (1, 1): "xy",
                        (3, 2): "x³y²", (4, 4): "x⁴y⁴"}.items():
        ax.text(i + 0.1, j + 0.12, lab, fontsize=8, color=INK)
    ax.set_title("lattice → group\n$G=\\mathbb{Z}_{L_x}\\times\\mathbb{Z}_{L_y}$", fontweight="normal")

    ax = axes[1]
    lattice(ax)
    ax.add_patch(plt.Polygon([(1, 1), (2, 1), (1, 2)], color=ORANGE, alpha=0.18, zorder=2))
    for p in [(1, 1), (2, 1), (1, 2)]:
        spin(ax, *p)
    ax.set_title("interaction → polynomial\n$Z_rZ_{r+\\hat x}Z_{r+\\hat y}\\ \\leftrightarrow\\ f=1+x+y$",
                 fontweight="normal")

    ax = axes[2]
    lattice(ax)
    for p in [(1, 1), (2, 1), (1, 2)]:
        ax.add_patch(plt.Circle(p, 0.26, fill=False, ls="--", color=ORANGE, zorder=3))
    ax.add_patch(plt.Polygon([(2, 2), (3, 2), (2, 3)], color=ORANGE, alpha=0.18, zorder=2))
    for p in [(2, 2), (3, 2), (2, 3)]:
        spin(ax, *p)
    ax.annotate("", (1.8, 1.8), (1.25, 1.25), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.text(0.55, 1.65, "$\\times xy$", fontsize=9)
    ax.set_title("translation → multiplication\n$O_{r+\\hat x+\\hat y}\\ \\leftrightarrow\\ xy\\,f$",
                 fontweight="normal")

    ax = axes[3]
    F = NEWMAN_MOORE.check_matrix(7)
    ax.imshow(F[:, 49:], cmap=ListedColormap(["white", INK]), interpolation="nearest")
    bare(ax)
    for s in ax.spines.values():
        s.set_visible(True)
    r = rank(F)
    ax.set_title(f"degeneracy → kernel\nNewman–Moore, 7×7: rank F = {r}, n = 49",
                 fontweight="normal")
    ax.set_xlabel(f"$\\log_2\\mathrm{{GSD}} = n-\\mathrm{{rank}}_{{\\mathbb{{F}}_2}}F = {49 - r}$")
    fig.tight_layout()
    save(fig, "pipeline.png")


# =================================================================== 2. three models
def draw_toric(ax):
    lattice(ax, 5)
    cx, cy = 1, 3                                   # star around vertex (1, 3)
    for dx, dy in [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]:
        ax.plot([cx, cx + 2 * dx], [cy, cy + 2 * dy], color=BLUE, lw=3, zorder=2)
        spin(ax, cx + dx, cy + dy, "X", BLUE)
    px, py = 2, 0                                   # plaquette with corner (2, 0)
    ax.add_patch(plt.Rectangle((px, py), 1, 1, fill=False, ec=ORANGE, lw=3, zorder=2))
    for dx, dy in [(0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)]:
        spin(ax, px + dx, py + dy, "Z", ORANGE)


def draw_nm(ax):
    lattice(ax, 5)
    for base in [(1, 1), (2, 3), (0, 3)]:
        tri = [base, (base[0] + 1, base[1]), (base[0], base[1] + 1)]
        ax.add_patch(plt.Polygon(tri, color=ORANGE, alpha=0.15, zorder=2))
        for p in tri:
            spin(ax, *p)


def draw_cube(ax, offset, terms, color):
    """Haah cube: vertex v of the unit cube carries the Paulis listed in terms[v]."""
    ox, oy = offset
    proj = lambda x, y, z: (ox + x + 0.45 * y, oy + z + 0.35 * y)            # noqa: E731
    edges = [((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)),
             ((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (1, 1, 0)),
             ((0, 1, 0), (0, 1, 1)), ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1)),
             ((1, 1, 0), (1, 1, 1)), ((1, 0, 1), (1, 1, 1)), ((0, 1, 1), (1, 1, 1))]
    for a, b in edges:
        ax.plot(*zip(proj(*a), proj(*b)), color=MUTED, lw=1, zorder=1)
    for v, lab in terms.items():
        x, y = proj(*v)
        ax.add_patch(plt.Rectangle((x - 0.17, y - 0.12), 0.34, 0.24, color=color, zorder=3))
        ax.text(x, y, lab, color="white", fontsize=6.5, ha="center", va="center", fontweight="bold",
                zorder=4)


def haah_cube_labels(gen, sector):
    """Which Pauli sits on which corner: corner exponent -> e.g. 'XI', 'IX', 'XX'."""
    first, second = (Poly(f) for f in gen)
    corners = {}
    for k, f in enumerate((first, second)):
        for e in f.terms:
            v = tuple(a + 1 for a in e) if sector == "Z" else e   # Z cube hangs off (1,1,1)
            lab = corners.get(v, ["I", "I"])
            lab[k] = sector
            corners[v] = lab
    return {v: "".join(lab) for v, lab in corners.items()}


def fig_models():
    Ls2 = np.arange(2, 33)
    data = {
        "toric": [TORIC_CODE.log_gsd(L) for L in Ls2],
        "nm": [NEWMAN_MOORE.log_gsd(L) for L in Ls2],
        "haah": [HAAH_CUBIC.log_gsd(L) for L in Ls2],
    }
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=0.35, wspace=0.18)
    titles = [
        ("Toric code", "$A_v=(1+\\bar x,\\ 1+\\bar y)$   $B_p=(1+y,\\ 1+x)$"),
        ("Newman–Moore", "$f = 1+x+y$   (classical, Z terms only)"),
        ("Haah's cubic code", "$(1+xy+yz+zx,\\ 1+x+y+z)$\n$(1+\\bar x+\\bar y+\\bar z,\\ 1+\\bar x\\bar y+\\bar y\\bar z+\\bar z\\bar x)$"),
    ]
    for k, (t, poly) in enumerate(titles):
        ax = fig.add_subplot(gs[0, k])
        if k == 0:
            draw_toric(ax)
        elif k == 1:
            draw_nm(ax)
        else:
            ax.set_xlim(-0.4, 3.9), ax.set_ylim(-0.6, 1.7), ax.set_aspect("equal"), bare(ax)
            draw_cube(ax, (0, 0), haah_cube_labels(("1 + xy + yz + zx", "1 + x + y + z"), "X"), BLUE)
            draw_cube(ax, (2.1, 0), haah_cube_labels(("1 + x^-1 + y^-1 + z^-1", "1 + x^-1y^-1 + y^-1z^-1 + z^-1x^-1"), "Z"), ORANGE)
        ax.set_title(t)
        ax.text(0.5, -0.04, poly, transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK)

    for k, key in enumerate(["toric", "nm", "haah"]):
        ax = fig.add_subplot(gs[1, k])
        ax.bar(Ls2, data[key], color=INK, width=0.6)
        ax.set_xlabel("$L$")
        ax.set_xticks([2, 8, 16, 24, 32])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 130)
        if k == 0:
            ax.set_ylabel("$\\log_2\\mathrm{GSD}$")
            ax.text(17, 12, "GSD = 4 for every L", ha="center", color=INK)
        if k == 1:
            Ls = np.array([3, 7, 15, 31])
            ax.plot(Ls2, Ls2 - 1, ls="--", color=MUTED, lw=1)
            ax.text(10, 34, "$L-1$ at $L=2^n-1$", color=INK)
        if k == 2:
            ax.plot(Ls2, 4 * Ls2 - 2, ls="--", color=MUTED, lw=1)
            ax.text(3, 100, "$4L-2$ at $L=2^n$", color=INK)
        ax.set_title(["$L\\times L$ torus", "$L\\times L$ torus", "$L\\times L\\times L$ torus"][k],
                     fontweight="normal", fontsize=9)
    save(fig, "models.png")


# =================================================================== 3. fractal symmetries
def fig_fractals():
    fig = plt.figure(figsize=(11, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.08)

    # closed: on the 63-torus x^64 = x, so fbar * (1 + fbar^63) = fbar + fbar^64 = 0
    L = 63
    fbar = Poly("1 + x^-1 + y^-1")
    u = ring_pow(fbar.on((L, L)), 63)
    u[0, 0] ^= 1
    F = NEWMAN_MOORE.check_matrix(L)
    assert not (F[:, L * L:] @ u.reshape(-1) % 2).any()
    ax = fig.add_subplot(gs[0])
    ax.imshow(u.T, cmap=TEALS, origin="lower", interpolation="nearest")
    for s in ax.spines.values():
        s.set_visible(True)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"closed on the 63×63 torus\nF s = 0: a symmetry ({NEWMAN_MOORE.log_gsd(L)} independent)",
                 fontweight="normal")

    # open: fbar^31 has 3^5 terms and violates only fbar^32 = 3 corners
    patt, corners = fractals.power_pattern(fbar, 5)
    ax = fig.add_subplot(gs[1])
    pts = np.array(list(patt.terms))
    c = np.array(list(corners.terms))
    ax.scatter(pts[:, 0], pts[:, 1], s=7, marker="s", color=TEAL, lw=0)
    ax.scatter(c[:, 0], c[:, 1], s=90, marker="*", color="#c81e3c", zorder=3)
    ax.set_aspect("equal"), bare(ax)
    ax.set_title(f"open: $\\bar f^{{\\,31}}$, {len(patt)} = 3⁵ flipped spins\nonly {len(corners)} violated terms, at the corners",
                 fontweight="normal")

    # Haah: Z on qubit 2 in the pattern sbar^31; X cubes see sbar^32 = 4 corners
    sbar = Poly("1 + x^-1 + y^-1 + z^-1")
    patt, corners = fractals.power_pattern(sbar, 5)
    ax = fig.add_subplot(gs[2], projection="3d")
    pts = np.array(list(patt.terms))
    c = np.array(list(corners.terms))
    ax.scatter(*pts.T, s=3, color=TEAL, depthshade=False, alpha=0.8)
    ax.scatter(*c.T, s=90, marker="*", color="#c81e3c", depthshade=False)
    ax.view_init(elev=22, azim=35)
    ax.set_axis_off()
    ax.set_title(f"open in Haah's code: {len(patt)} = 4⁵ Z's\nonly {len(corners)} excitations: fractons",
                 fontweight="normal", y=0.98)
    save(fig, "fractals.png")


# =================================================================== 4. UV/IR mixing
def fig_uv_ir():
    Ls = np.arange(2, 65)
    g = np.array([[NEWMAN_MOORE.log_gsd((lx, ly)) for lx in Ls] for ly in Ls], dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eef0f3")
    shown = np.where(g > 0, np.log2(np.maximum(g, 1)) + 1, np.nan)
    im = ax.imshow(shown, origin="lower", cmap=cmap, interpolation="nearest",
                   extent=[1.5, 64.5, 1.5, 64.5], vmin=1, vmax=np.log2(g.max()) + 1)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    ticks = [1, 2, 4, 8, 16, 32, 62]
    cb.set_ticks(np.log2(ticks) + 1, labels=[str(t) for t in ticks])
    cb.set_label("$\\log_2\\mathrm{GSD}$")
    cb.outline.set_visible(False)
    ax.set_xlabel("$L_x$"), ax.set_ylabel("$L_y$")
    ax.set_xticks([2, 16, 32, 48, 64]), ax.set_yticks([2, 16, 32, 48, 64])
    ax.set_title("Newman–Moore on $L_x\\times L_y$ tori\ngray: unique ground state", fontweight="normal")
    save(fig, "uv_ir.png")


# =================================================================== 5. stencils as cellular automata
def fig_automata():
    hz = classical("haah_z", STENCILS["haah_z"])
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, exps, t in [(axes[0], (0, 1), "Newman–Moore CA\n$t_{r+1}=(1+x)\\,t_r$"),
                        (axes[1], (0, 1, 2), "trinomial CA\n$t_{r+1}=(1+x+x^2)\\,t_r$")]:
        ax.imshow(fractals.ca_rows(exps, 64)[:, 64:], cmap=TEALS, origin="lower",
                  interpolation="nearest", aspect="auto")
        bare(ax)
        ax.set_title(t, fontweight="normal")
    for ax, L in [(axes[2], 32), (axes[3], 63)]:
        S = hz.symmetries(L)
        v = S[int(np.argmin(S.sum(1)))]
        ax.imshow(v.reshape(L, L).T, cmap=TEALS, origin="lower", interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"haah_z symmetry, {L}×{L}\n{len(S)} independent", fontweight="normal")
    fig.tight_layout()
    save(fig, "automata.png")


FIGURES = {"pipeline": fig_pipeline, "models": fig_models, "fractals": fig_fractals,
           "uv_ir": fig_uv_ir, "automata": fig_automata}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    wanted = sys.argv[1:]
    for name, make in FIGURES.items():
        if not wanted or any(w in name for w in wanted):
            make()
