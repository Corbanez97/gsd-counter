"""Fractal subsystem symmetries: where the kernel vectors get their shape.

Two facts do all the work.

* **A stencil is a cellular automaton.** For f = y + g(x), the constraint f * t = 0
  says row r+1 = g(x) * row r. Iterating from a single seed draws the light cone
  of the rule (Pascal's triangle mod 2 for Newman-Moore). On the torus the CA
  has to close up on itself, and its closed orbits are the kernel vectors.

* **Frobenius.** Over F_2, (a + b)^2 = a^2 + b^2, so f^(2^n) = f(x^(2^n), y^(2^n)):
  a power of the stencil is the stencil itself, blown up by 2^n. Flipping the
  spins on u = fbar^(2^n - 1) therefore violates only the terms
  fbar * u = fbar^(2^n): one per term of f, at the corners of a fractal. That is
  an *open* symmetry operator, and its corners are the fracton excitations.
"""

from __future__ import annotations

import numpy as np

from .ring import Poly

__all__ = ["ca_rows", "power_pattern", "truncate", "to_grid"]


def ca_rows(g_exponents=(0, 1), nrows=64, width=None):
    """Rows of the CA  t_{r+1} = g(x) t_r  from a single seed (a 0/1 array, nrows x width)."""
    width = width or 2 * nrows + 3
    out = np.zeros((nrows, width), dtype=np.uint8)
    row = np.zeros(width, dtype=np.uint8)
    row[width // 2] = 1
    for r in range(nrows):
        out[r] = row
        nxt = np.zeros(width, dtype=np.uint8)
        for e in g_exponents:
            nxt[e:] ^= row[: width - e] if e else row
        row = nxt
    return out


def power_pattern(f, n, p=2):
    """(u, f * u) with u = f^(p^n - 1): a fractal operator and its corner excitations."""
    u = f.pow(p ** n - 1, p)
    return u, (f * u).mod(p)


def to_grid(poly, pad=1):
    """Draw a sparse Poly on the smallest array containing it. Returns (array, origin)."""
    exps = np.array(list(poly.terms))
    lo = exps.min(axis=0) - pad
    shape = tuple(exps.max(axis=0) - lo + 1 + pad)
    a = np.zeros(shape, dtype=np.int64)
    for e, c in poly.terms.items():
        a[tuple(np.array(e) - lo)] = c
    return a, tuple(-lo)


def truncate(u, F, region, p=2):
    """Keep a closed symmetry generator only inside ``region``; return (patch, violated terms).

    A closed kernel vector commutes with every term. Cut it, and the only terms it
    still violates sit on the cut: the open operator is a membrane whose
    boundary carries the excitations.
    """
    patch = (np.asarray(u) * np.asarray(region).reshape(np.shape(u))) % p
    return patch, (F @ patch.reshape(-1)) % p

