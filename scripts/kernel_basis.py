#!/usr/bin/env python3
"""N_c = dim ker A for a stencil operator on an Ly x Lx torus over GF(p), plus
one image per kernel basis vector (each is a "fractal" ground-state / relation
pattern folded back onto the Ly x Lx lattice).

Two outputs per size, both written under a subfolder named after the operator,
size and modulus:
    1. images/  -- one PNG per basis vector of ker A
    2. summary.txt -- Lx, Ly, p, N_c, GSD = p**N_c

Sizes: give one L for a square lattice (Lx = Ly = L), or two (Lx then Ly) for
a rectangular torus.

Examples
--------
    python3 scripts/kernel_basis.py 21                  # haah_z, p=2, 21x21
    python3 scripts/kernel_basis.py 12 8                 # Lx=12, Ly=8
    python3 scripts/kernel_basis.py 21 --op six -p 3
    python3 scripts/kernel_basis.py 21 42 85 --op haah_z # sweep several sizes
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nc_sweep import OPS, build_matrix


# ------------------------------------------------------- GF(2): bit-packed RREF
def rref_gf2(A):
    """Full RREF of a square 0/1 matrix over GF(2), bit-packed into uint64 words.

    Returns (M, pivot_cols): M is the reduced matrix as packed words (same
    layout produced by np.packbits(..., bitorder='little') then viewed as
    uint64), pivot_cols[i] is the pivot column of row i of M.
    """
    rows, cols = A.shape
    M = np.packbits(np.asarray(A & 1, dtype=np.uint8), axis=1, bitorder="little")
    M = np.pad(M, ((0, 0), (0, (-M.shape[1]) % 8))).view(np.uint64)
    pivot_cols = []
    r = 0
    for col in range(cols):
        if r == rows:
            break
        word, bit = col >> 6, np.uint64(col & 63)
        colbits = (M[r:, word] >> bit) & np.uint64(1)
        nz = np.flatnonzero(colbits)
        if nz.size == 0:
            continue
        piv = r + int(nz[0])
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        hit = np.flatnonzero((M[:, word] >> bit) & np.uint64(1))
        hit = hit[hit != r]
        if hit.size:
            M[hit] ^= M[r]
        pivot_cols.append(col)
        r += 1
    return M, pivot_cols


def nullspace_gf2(M, pivot_cols, n_cols):
    """Kernel basis from a bit-packed RREF: one vector per free column."""
    free_cols = [c for c in range(n_cols) if c not in pivot_cols]
    basis = np.zeros((len(free_cols), n_cols), dtype=np.uint8)
    for k, f in enumerate(free_cols):
        basis[k, f] = 1
        word, bit = f >> 6, np.uint64(f & 63)
        bits = (M[: len(pivot_cols), word] >> bit) & np.uint64(1)
        for i, pc in enumerate(pivot_cols):
            basis[k, pc] = int(bits[i])
    return basis


# ------------------------------------------------------- GF(p), p odd prime
def rref_gfp(A, p):
    """Full RREF over GF(p), plain vectorised elimination (small matrices)."""
    M = A.astype(np.int64) % p
    rows, cols = M.shape
    pivot_cols = []
    r = 0
    for col in range(cols):
        if r == rows:
            break
        nz = np.flatnonzero(M[r:, col])
        if nz.size == 0:
            continue
        piv = r + int(nz[0])
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, col]), p - 2, p)) % p
        factors = M[:, col].copy()
        factors[r] = 0
        M = (M - np.outer(factors, M[r])) % p
        pivot_cols.append(col)
        r += 1
    return M, pivot_cols


def nullspace_gfp(M, pivot_cols, n_cols, p):
    free_cols = [c for c in range(n_cols) if c not in pivot_cols]
    basis = np.zeros((len(free_cols), n_cols), dtype=np.int64)
    for k, f in enumerate(free_cols):
        basis[k, f] = 1
        for i, pc in enumerate(pivot_cols):
            basis[k, pc] = (-int(M[i, f])) % p
    return basis


def kernel_basis(shifts, Lx, Ly, p):
    A = build_matrix(shifts, Lx, Ly, p)
    if p == 2:
        M, pivot_cols = rref_gf2(A)
        basis = nullspace_gf2(M, pivot_cols, Lx * Ly)
    else:
        M, pivot_cols = rref_gfp(A, p)
        basis = nullspace_gfp(M, pivot_cols, Lx * Ly, p)
    return basis


# ------------------------------------------------------------------ rendering
def render_vector(v, Ly, Lx, p, path, target_px=512):
    """Save one basis vector, folded onto the Ly x Lx lattice, as a PNG.

    Block-replicated (nearest-neighbour) rather than resampled, so the fractal
    structure stays crisp instead of blurring away on small lattices.
    """
    grid = v.reshape(Ly, Lx)
    scale = max(1, target_px // max(Ly, Lx))
    if scale > 1:
        grid = np.kron(grid, np.ones((scale, scale), dtype=grid.dtype))
    plt.imsave(path, grid, cmap="Purples", vmin=0, vmax=max(1, p - 1), origin="upper")


# ------------------------------------------------------------------ driver
def run(op_name, Lx, Ly, p, outdir):
    shifts = OPS[op_name]
    basis = kernel_basis(shifts, Lx, Ly, p)
    Nc = basis.shape[0]

    if Lx == Ly:
        tag = f"{op_name}_L{Lx}_p{p}"
    else:
        tag = f"{op_name}_Lx{Lx}_Ly{Ly}_p{p}"
    folder = os.path.join(outdir, tag)
    images_dir = os.path.join(folder, "images")
    os.makedirs(images_dir, exist_ok=True)

    for idx, v in enumerate(basis):
        path = os.path.join(images_dir, f"{tag}_v{idx:03d}.png")
        render_vector(v, Ly, Lx, p, path)

    summary_path = os.path.join(folder, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write(f"op={op_name}\n")
        fh.write(f"Lx={Lx}\n")
        fh.write(f"Ly={Ly}\n")
        fh.write(f"p={p}\n")
        fh.write(f"Nc={Nc}\n")
        fh.write(f"GSD={p}**{Nc} = {p ** Nc}\n")

    print(f"{op_name} p={p}  Lx={Lx:4d} Ly={Ly:4d}  Nc={Nc:4d}  "
          f"GSD={p}^{Nc}  -> {folder}  ({Nc} images)")
    return Nc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sizes", type=int, nargs="+",
                    help="one L (square lattice) or two (Lx then Ly); "
                         "more than two are treated as a sweep of square sizes")
    ap.add_argument("--op", default="haah_z", choices=sorted(OPS))
    ap.add_argument("-p", type=int, default=2, help="prime modulus (default 2)")
    ap.add_argument("-o", "--outdir", default="data/kernel_basis",
                    help="output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if len(args.sizes) == 2:
        pairs = [(args.sizes[0], args.sizes[1])]
    else:
        pairs = [(L, L) for L in args.sizes]

    for Lx, Ly in pairs:
        run(args.op, Lx, Ly, args.p, args.outdir)


if __name__ == "__main__":
    main()
