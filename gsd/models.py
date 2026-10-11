"""Translation-invariant stabilizer models and their ground-state degeneracy.

A model has ``q`` qudits per site and a few generator *species*. Each species is
one Pauli operator at the origin, written as polynomials: its X part
``(X_1, ..., X_q)`` and Z part ``(Z_1, ..., Z_q)`` say which qudit of which
neighbouring site it touches. Translating it to every site of the torus gives
all the terms of the Hamiltonian

    H = - sum_r sum_species  O_species(r).

The check matrix F has one row per term and one column per qudit
(``[X block | Z block]``, length 2n). Commuting terms that square to one share
a ground space where every O_r = +1, and each independent term halves it:

    log_p GSD = n - rank F.

Every relation  prod_r O_r^{t_r} = 1  is a vector in the left kernel of F and
lowers the rank by one; that is where all the degeneracy beyond 2^0 comes from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linalg import nullspace, rank
from .ring import Poly, mult_matrix, quotient_dim

__all__ = ["Generator", "Model", "classical", "MODELS", "STENCILS"]


@dataclass(frozen=True)
class Generator:
    """One species of stabilizer: X and Z parts, one polynomial per qudit of the unit cell."""

    x: tuple
    z: tuple

    @property
    def sector(self):
        """'X' or 'Z' for CSS-type generators, None for mixed ones."""
        has_x = any(len(f) for f in self.x)
        has_z = any(len(f) for f in self.z)
        return "X" if has_x and not has_z else "Z" if has_z and not has_x else None


@dataclass(frozen=True)
class Model:
    name: str
    dim: int
    qudits: int
    generators: tuple
    description: str = ""

    # ------------------------------------------------------------ matrices
    def check_matrix(self, L, p=2):
        """F: one row per term O_r, columns = [X on every qudit | Z on every qudit].

        Qudit (site r, species a) has column a*N + flat(r).
        """
        shape = _shape(L, self.dim)
        blocks = []
        for g in self.generators:
            # row r of M(f)^T is the stencil f placed at site r
            xs = [mult_matrix(f, shape, p).T for f in g.x]
            zs = [mult_matrix(f, shape, p).T for f in g.z]
            blocks.append(np.concatenate(xs + zs, axis=1))
        return np.concatenate(blocks, axis=0).astype(np.int64)

    def commutes(self, L, p=2):
        """True iff every pair of terms commutes: F Lambda F^T = 0 (symplectic form)."""
        F = self.check_matrix(L, p)
        n = F.shape[1] // 2
        X, Z = F[:, :n], F[:, n:]
        return not ((X @ Z.T - Z @ X.T) % p).any()

    # ------------------------------------------------------------ degeneracy
    def n_qudits(self, L):
        return self.qudits * int(np.prod(_shape(L, self.dim)))

    def log_gsd(self, L, p=2, method="auto"):
        """log_p of the ground-state degeneracy on the torus of side(s) L.

        ``method="brute"``  n - rank F, by Gaussian elimination on the check matrix.
        ``method="algebra"`` the same number from the group ring: a single species
            with components (f_1, ..., f_m) has rank F_species = dim (f_1, ..., f_m)
            = N - dim R/(f_1, ..., f_m). (Group rings are Frobenius algebras, so the
            annihilator of an ideal I has dimension dim R/I.) Valid when no two
            species can cancel against each other: at most one X-type and one
            Z-type species. That covers every model in ``MODELS``.
        ``method="auto"`` algebra when valid, else brute.
        """
        shape = _shape(L, self.dim)
        sectors = [g.sector for g in self.generators]
        algebra_ok = None not in sectors and len(set(sectors)) == len(sectors)
        if method == "algebra" and not algebra_ok:
            raise ValueError(f"{self.name}: algebraic shortcut needs <= 1 species per sector")
        if method == "brute" or not algebra_ok:
            return self.n_qudits(L) - rank(self.check_matrix(L, p), p)
        N = int(np.prod(shape))
        n_rank = 0
        for g in self.generators:
            comps = [f for f in g.x + g.z if len(f)]
            n_rank += N - quotient_dim(comps, shape, p)
        return self.n_qudits(L) - n_rank

    def gsd(self, L, p=2):
        return p ** self.log_gsd(L, p)

    # ------------------------------------------------------------ kernels
    def symmetries(self, L, p=2):
        """Pauli operators commuting with every term, one per row.

        Classical models (Z terms only): the spin-flip patterns u with F_Z u = 0,
        length n. Otherwise the whole commutant {s : F Lambda s = 0} as symplectic
        vectors of length 2n (it contains the stabilizers and the logical operators).
        """
        F = self.check_matrix(L, p)
        n = F.shape[1] // 2
        if not F[:, :n].any():
            return nullspace(F[:, n:], p)
        return nullspace(np.concatenate([F[:, n:], -F[:, :n]], axis=1) % p, p)

    def relations(self, L, p=2):
        """Basis of ker F^T: exponents t with prod_r O_r^{t_r} = 1, one per row."""
        return nullspace(self.check_matrix(L, p).T, p)

    def __repr__(self):
        return f"Model({self.name!r})"


def _shape(L, dim):
    return tuple(L) if np.iterable(L) else (int(L),) * dim


# ====================================================================== constructors
def classical(name, f, description=""):
    """A classical spin model H = -sum_r prod_{s in f} sigma_{r+s}: one Z-type stencil."""
    f = Poly(f) if isinstance(f, str) else f
    return Model(name, f.dim, 1, (Generator(x=(Poly(0, f.dim),), z=(f,)),), description)


def _css(name, dim, xs, zs, description):
    P = lambda s: Poly(s, dim=dim)                                        # noqa: E731
    zero = tuple(P("0") for _ in xs)
    return Model(name, dim, len(xs),
                 (Generator(tuple(map(P, xs)), zero), Generator(zero, tuple(map(P, zs)))),
                 description)


# The three models of the poster. Signs make the toric code valid over any Z_p;
# at p = 2 they read exactly as A_v = (1 + x^-1, 1 + y^-1), B_p = (1 + y, 1 + x).
TORIC_CODE = _css(
    "toric_code", 2,
    xs=["1 - x^-1", "1 - y^-1"],
    zs=["1 - y", "x - 1"],
    description="Kitaev's toric code: qubits on the two edges of each unit cell.",
)
NEWMAN_MOORE = classical(
    "newman_moore", "1 + x + y",
    "Newman-Moore: classical three-spin interaction on a triangle.",
)
HAAH_CUBIC = _css(
    "haah_cubic", 3,
    xs=["1 + xy + yz + zx", "1 + x + y + z"],
    zs=["1 + x^-1 + y^-1 + z^-1", "1 + x^-1y^-1 + y^-1z^-1 + z^-1x^-1"],
    description="Haah's cubic code: two qubits per site, X and Z cubes.",
)

MODELS = {m.name: m for m in (TORIC_CODE, NEWMAN_MOORE, HAAH_CUBIC)}

# Single-stencil classical models explored in the notebooks.
STENCILS = {
    "newman_moore": "1 + x + y",
    "haah_z":       "1 + x + x^2 + y + xy + y^2",   # 2D corner stencil, closed form in closed_form.py
    "haah_x":       "1 + x^-1 + x^-2 + y^-1 + x^-1y^-1 + y^-2",   # its reversal (transpose)
    "trinomial":    "1 + x + x^2 + y",
    "six":          "x + x^-1 + y + y^-1 + x^-1y + xy^-1",          # triangular-lattice neighbours
    "two_roll":     "1 + y + x^-1",
}
STENCILS = {k: Poly(v) for k, v in STENCILS.items()}
