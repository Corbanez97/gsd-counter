#!/usr/bin/env python3
"""N_c = dim ker A for a stencil operator on an Ly x Lx torus over GF(p).

Writes one CSV per operator, named after it (e.g. haah_z.csv), with columns
Lx, Ly, Nc.  GSD = p ** Nc.

Examples
--------
    python3 scripts/nc_sweep.py                       # haah_z, p=2, L = 2..23 (square)
    python3 scripts/nc_sweep.py --lx 2-40 --ly 2-40   # full rectangular grid
    python3 scripts/nc_sweep.py --lx 8                 # Lx=8, Ly sweeps 2..8
    python3 scripts/nc_sweep.py --lx 4 --ly 2-64      # fixed Lx, explicit Ly range
    python3 scripts/nc_sweep.py --op six -p 3
"""

import argparse
import csv
import os

import numpy as np

# ---------------------------------------------------------------- operators
# shifts are (d_row, d_col) = (e_2, e_1); the lattice is periodic in both.
HAAH_Z_SHIFTS = [
    (0, 0), (0, 1), (0, 2),   # bottom row: 3 cells
    (1, 0), (1, 1),           # middle row: 2 cells
    (2, 0),                   # top row:    1 cell
]

OPS = {
    "haah_z":   HAAH_Z_SHIFTS,
    "haah_x":   [(-d2, -d1) for (d2, d1) in HAAH_Z_SHIFTS],
    "six":      [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, 1), (1, -1)],
    "two_roll": [(0, 0), (1, 0), (0, -1)],
    "nm":       [(0, 0), (0, 1), (1, 0)],
}


def build_matrix(shifts, Lx, Ly, p=2):
    """Matrix of the operator on an Ly x Lx torus: column k = op(delta_k)."""
    N = Lx * Ly
    dtype = np.uint8 if p < 128 else np.int64   # keeps big lattices in memory
    A = np.zeros((N, N), dtype=dtype)
    r, c = np.divmod(np.arange(N), Lx)
    for term in shifts:
        d2, d1, coeff = (term + (1,))[:3] if len(term) == 2 else term
        rows = ((r + d2) % Ly) * Lx + (c + d1) % Lx
        A[rows, np.arange(N)] += coeff
    return A % p


# ------------------------------------------------------------ rank over GF(p)
def rank_gf2(A):
    """Rank over GF(2) with rows bit-packed into uint64 words."""
    rows, cols = A.shape
    M = np.packbits(np.asarray(A & 1, dtype=np.uint8), axis=1, bitorder="little")
    M = np.pad(M, ((0, 0), (0, (-M.shape[1]) % 8))).view(np.uint64)
    r = 0
    for col in range(cols):
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
        M[hit] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def rank_gfp(A, p):
    """Rank over GF(p), p prime; plain vectorised elimination."""
    if p == 2:
        return rank_gf2(A)
    M = A.astype(np.int64) % p
    rows, cols = M.shape
    r = 0
    for col in range(cols):
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
        r += 1
        if r == rows:
            break
    return r


def Nc(shifts, Lx, Ly, p=2):
    return Lx * Ly - rank_gfp(build_matrix(shifts, Lx, Ly, p), p)


# ------------------------------------------------------------------ driver
def parse_range(spec):
    """'2-23' -> [2..23];  '8' -> [8];  '2,4,8' -> [2, 4, 8]."""
    out = []
    for part in spec.split(","):
        if "-" in part.strip("-"):
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", default="haah_z", choices=sorted(OPS))
    ap.add_argument("-p", type=int, default=2, help="prime modulus (default 2)")
    ap.add_argument("--lx", default="2-128", help="Lx values, e.g. 2-23 or 4 or 2,4,8")
    ap.add_argument("--ly", default=None,
                    help="Ly values; default: 2..Lx for a single Lx, else same as --lx")
    ap.add_argument("--rect", action="store_true",
                    help="full Lx x Ly grid (default: only Lx == Ly)")
    ap.add_argument("-o", "--outdir", default="data", help="output directory")
    args = ap.parse_args()

    shifts = OPS[args.op]
    lxs = parse_range(args.lx)
    if args.ly:
        lys = parse_range(args.ly)
    elif len(lxs) == 1:
        lys = list(range(2, lxs[0] + 1))   # a single Lx sweeps Ly = 2..Lx
    else:
        lys = list(lxs)

    if args.rect or len(lxs) == 1 or len(lys) == 1:
        pairs = [(lx, ly) for ly in lys for lx in lxs]
    else:
        pairs = [(L, L) for L in lxs if L in set(lys)]

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.op}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Lx", "Ly", "Nc"])
        for lx, ly in pairs:
            n = Nc(shifts, lx, ly, args.p)
            w.writerow([lx, ly, n])
            fh.flush()
            print(f"{args.op} p={args.p}  Lx={lx:3d} Ly={ly:3d}  Nc={n:4d}", flush=True)
    print(f"\nwrote {len(pairs)} rows to {path}")


if __name__ == "__main__":
    main()
