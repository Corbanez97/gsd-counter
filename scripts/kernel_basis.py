#!/usr/bin/env python3
"""Draw every independent symmetry of a classical stencil model as a PNG.

Each image is one basis vector of ker F (a spin-flip pattern that commutes with
every term), folded back onto the Lx x Ly torus. For fractal stencils these are
the fractal subsystem symmetries. One folder per size:

    data/kernel_basis/<model>_L<L>_p<p>/{images/*.png, summary.txt}

    python scripts/kernel_basis.py 21                       # haah_z, 21 x 21
    python scripts/kernel_basis.py 12 8 --model trinomial   # Lx = 12, Ly = 8
    python scripts/kernel_basis.py 21 42 85                 # several square sizes
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gsd import STENCILS, classical  # noqa: E402


def render(v, shape, p, path, target_px=512):
    """x to the right, y up; block-replicated so small lattices stay crisp."""
    grid = v.reshape(shape).T
    scale = max(1, target_px // max(shape))
    grid = np.kron(grid, np.ones((scale, scale), dtype=grid.dtype))
    plt.imsave(path, grid, cmap="Greens", vmin=0, vmax=max(1, p - 1), origin="lower")


def run(name, shape, p, outdir):
    model = classical(name, STENCILS[name])
    basis = model.symmetries(shape, p)
    tag = f"{name}_L{shape[0]}" if shape[0] == shape[1] else f"{name}_Lx{shape[0]}_Ly{shape[1]}"
    folder = os.path.join(outdir, f"{tag}_p{p}")
    os.makedirs(os.path.join(folder, "images"), exist_ok=True)
    for i, v in enumerate(basis):
        render(v, shape, p, os.path.join(folder, "images", f"{tag}_p{p}_v{i:03d}.png"))
    with open(os.path.join(folder, "summary.txt"), "w") as fh:
        fh.write(f"model={name}\nf={STENCILS[name]}\nLx={shape[0]}\nLy={shape[1]}\np={p}\n"
                 f"log_gsd={len(basis)}\n")
    print(f"{name} p={p}  {shape[0]}x{shape[1]}  log_p GSD = {len(basis)}  -> {folder}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sizes", type=int, nargs="+",
                    help="one L (square), two (Lx Ly), or more (several square sizes)")
    ap.add_argument("--model", default="haah_z", choices=sorted(STENCILS))
    ap.add_argument("-p", type=int, default=2)
    ap.add_argument("-o", "--outdir", default="data/kernel_basis")
    args = ap.parse_args()
    shapes = [tuple(args.sizes)] if len(args.sizes) == 2 else [(L, L) for L in args.sizes]
    for shape in shapes:
        run(args.model, shape, args.p, args.outdir)


if __name__ == "__main__":
    main()
