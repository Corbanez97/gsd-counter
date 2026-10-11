"""Linear algebra over the finite field GF(p): rank, RREF and null space.

Ordinary ``numpy.linalg`` works over the reals and gives the wrong answer here.
Gaussian elimination over GF(p) is the textbook algorithm with two changes:
arithmetic is mod p, and a pivot is normalised by its modular inverse
(over GF(2) the only nonzero scalar is 1, so that step disappears, and
"subtract a row" is the same as "add a row", i.e. XOR).

For p = 2 the rows are bit-packed into 64-bit words, so one XOR handles 64
matrix entries at once. That is what makes 4096 x 4096 matrices cheap.
A deliberately slow, readable version of the same algorithm is worked through
by hand in ``notebooks/01_gf2_linear_algebra.ipynb``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rank", "rref", "nullspace"]


# ---------------------------------------------------------------- GF(2), packed
def _pack(A):
    """Pack a 0/1 matrix row-wise into uint64 words (bit j of word w = column 64w + j)."""
    M = np.packbits(np.asarray(A, dtype=np.uint8) & 1, axis=1, bitorder="little")
    M = np.pad(M, ((0, 0), (0, (-M.shape[1]) % 8)))
    return np.ascontiguousarray(M).view(np.uint64)


def _unpack(M, n_cols):
    bits = np.unpackbits(M.view(np.uint8), axis=1, bitorder="little")
    return bits[:, :n_cols]


def _eliminate_gf2(A, reduced):
    """Row-reduce over GF(2). Returns (packed matrix, pivot columns)."""
    rows, cols = A.shape
    M = _pack(A)
    pivots, r = [], 0
    for col in range(cols):
        if r == rows:
            break
        word, bit = col >> 6, np.uint64(col & 63)
        column = (M[:, word] >> bit) & np.uint64(1)
        below = np.flatnonzero(column[r:])
        if below.size == 0:
            continue                                   # free column
        piv = r + int(below[0])
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
            column[[r, piv]] = column[[piv, r]]
        # "reduced" clears the column above the pivot too; rank only needs below
        hit = np.flatnonzero(column if reduced else np.r_[np.zeros(r + 1, bool), column[r + 1:]])
        hit = hit[hit != r]
        if hit.size:
            M[hit] ^= M[r]
        pivots.append(col)
        r += 1
    return M, pivots


# ---------------------------------------------------------------- GF(p), p odd
def _eliminate_gfp(A, p):
    M = np.asarray(A, dtype=np.int64) % p
    rows, cols = M.shape
    pivots, r = [], 0
    for col in range(cols):
        if r == rows:
            break
        below = np.flatnonzero(M[r:, col])
        if below.size == 0:
            continue
        piv = r + int(below[0])
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, col]), p - 2, p)) % p       # Fermat inverse
        factors = M[:, col].copy()
        factors[r] = 0
        M = (M - np.outer(factors, M[r])) % p
        pivots.append(col)
        r += 1
    return M, pivots


# ---------------------------------------------------------------- public API
def rref(A, p=2):
    """Reduced row echelon form over GF(p).

    Returns ``(R, pivots)`` where ``pivots[i]`` is the pivot column of row ``i``.
    """
    A = np.asarray(A)
    if p == 2:
        M, pivots = _eliminate_gf2(A, reduced=True)
        return _unpack(M, A.shape[1]).astype(np.int64), pivots
    return _eliminate_gfp(A, p)


def rank(A, p=2):
    """Rank over GF(p)."""
    A = np.asarray(A)
    if A.size == 0:
        return 0
    if p == 2:
        return len(_eliminate_gf2(A, reduced=False)[1])
    return len(_eliminate_gfp(A, p)[1])


def nullspace(A, p=2):
    """Basis of ``{v : A v = 0}`` over GF(p), one vector per row.

    One basis vector per free (non-pivot) column f: set v[f] = 1, all other
    free entries 0, and solve each pivot entry from its RREF row,
    v[pivot_i] = -R[i, f]  (over GF(2) the minus sign is invisible).
    """
    A = np.asarray(A)
    n = A.shape[1]
    R, pivots = rref(A, p)
    free = np.setdiff1d(np.arange(n), pivots)
    basis = np.zeros((free.size, n), dtype=np.int64)
    basis[np.arange(free.size), free] = 1
    if pivots:
        basis[:, pivots] = (-R[: len(pivots)][:, free].T) % p
    return basis
