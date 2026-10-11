"""The group ring F_p[G] of a periodic lattice, and stencils as its elements.

The dictionary this module implements:

    lattice  Z_L1 x ... x Z_Ld           ->  group G
    pattern of numbers on the sites      ->  element of F_p[G] = F_p[x, y, ...]/(x^L1 - 1, ...)
    local stencil (which sites a term    ->  Laurent polynomial f, e.g. 1 + x + y
      touches, relative to its origin)
    translate the stencil by r           ->  multiply by the monomial x^r
    stamp it on every site and add       ->  multiply by f  (a circulant matrix)

A ``Poly`` is a Laurent polynomial written the way it is on paper, independent
of the lattice size. ``Poly.on(shape)`` turns it into a concrete group-ring
element: a numpy array with one entry per site, exponents read mod L.
Array axis k is variable k, so ``a[i, j]`` is the coefficient of x^i y^j.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .linalg import rank

__all__ = ["Poly", "ring_mul", "ring_pow", "mult_matrix", "quotient_dim"]

VARS = "xyzw"


# ====================================================================== Laurent polynomials
@dataclass(frozen=True)
class Poly:
    """Laurent polynomial in ``dim`` variables: ``{exponent tuple: coefficient}``.

    Build one from a string, ``Poly("1 + x + y")``, ``Poly("1 + x^-1 + y^-1")``,
    ``Poly("1 + xy + yz + zx")``, or ``Poly("1 - x", dim=2)``. Coefficients are
    integers; they are reduced mod p only when the polynomial meets a lattice.
    """

    terms: dict = field(default_factory=dict)
    dim: int = 2

    def __init__(self, spec="0", dim=None):
        if isinstance(spec, dict):
            terms = dict(spec)
            d = dim or (len(next(iter(terms))) if terms else 2)
        else:
            terms, d = _parse(str(spec))
            d = max(d, dim or 2)
            terms = {e + (0,) * (d - len(e)): c for e, c in terms.items()}
        object.__setattr__(self, "terms", {e: c for e, c in terms.items() if c})
        object.__setattr__(self, "dim", d)

    # ------------------------------------------------------------ algebra
    def __add__(self, other):
        out = dict(self.terms)
        for e, c in _as_poly(other, self.dim).terms.items():
            out[e] = out.get(e, 0) + c
        return Poly(out, self.dim)

    __radd__ = __add__

    def __neg__(self):
        return Poly({e: -c for e, c in self.terms.items()}, self.dim)

    def __sub__(self, other):
        return self + (-_as_poly(other, self.dim))

    def __mul__(self, other):
        other = _as_poly(other, self.dim)
        out = {}
        for e1, c1 in self.terms.items():
            for e2, c2 in other.terms.items():
                e = tuple(a + b for a, b in zip(e1, e2))
                out[e] = out.get(e, 0) + c1 * c2
        return Poly(out, self.dim)

    __rmul__ = __mul__

    def mod(self, p):
        """Reduce coefficients mod p."""
        return Poly({e: c % p for e, c in self.terms.items()}, self.dim)

    def pow(self, n, p=2):
        """f^n with coefficients mod p (sparse: no lattice needed)."""
        result, base = Poly({(0,) * self.dim: 1}, self.dim), self.mod(p)
        while n:
            if n & 1:
                result = (result * base).mod(p)
            base = (base * base).mod(p)
            n >>= 1
        return result

    def bar(self):
        """The antipode x -> x^-1: the stencil reflected through its origin."""
        return Poly({tuple(-a for a in e): c for e, c in self.terms.items()}, self.dim)

    def substitute(self, **scale):
        """Frobenius-style rescaling, e.g. ``f.substitute(x=2, y=2)`` = f(x^2, y^2)."""
        k = [scale.get(v, 1) for v in VARS[: self.dim]]
        return Poly({tuple(a * s for a, s in zip(e, k)): c for e, c in self.terms.items()}, self.dim)

    # ------------------------------------------------------------ on a lattice
    def on(self, shape, p=2):
        """The group-ring element: an array of ``shape`` with exponents taken mod L."""
        shape = tuple(shape)
        if len(shape) != self.dim:
            raise ValueError(f"{self} is {self.dim}-dimensional, lattice is {shape}")
        a = np.zeros(shape, dtype=np.int64)
        for e, c in self.terms.items():
            a[tuple(ei % L for ei, L in zip(e, shape))] += c
        return a % p

    def offsets(self):
        """The stencil as a list of (offset tuple, coefficient)."""
        return sorted(self.terms.items())

    def __len__(self):
        return len(self.terms)

    def __eq__(self, other):
        return isinstance(other, Poly) and self.dim == other.dim and self.terms == other.terms

    def __hash__(self):
        return hash((self.dim, tuple(sorted(self.terms.items()))))

    def __repr__(self):
        return f"Poly('{self}')"

    def __str__(self):
        if not self.terms:
            return "0"
        parts = []
        for e, c in sorted(self.terms.items(), key=lambda t: (sum(map(abs, t[0])), tuple(-abs(a) for a in t[0]), tuple(-a for a in t[0]))):
            mono = "".join(VARS[k] + (f"^{a}" if a != 1 else "") for k, a in enumerate(e) if a)
            if not mono:
                parts.append(str(c))
            else:
                parts.append(("" if c == 1 else "-" if c == -1 else str(c)) + mono)
        return " + ".join(parts).replace("+ -", "- ")


def _as_poly(obj, dim):
    if isinstance(obj, Poly):
        return obj
    return Poly({(0,) * dim: int(obj)}, dim)


_TERM = re.compile(r"([xyzw])(?:\^\(?(-?\d+)\)?)?")


def _parse(s):
    s = s.replace(" ", "").replace("*", "")
    if not s:
        return {}, 2
    terms, dim = {}, 1
    for chunk in re.split(r"(?<![\^(])(?=[+-])", s):         # split at +/- not in an exponent
        if not chunk:
            continue
        sign, body = (chunk[0], chunk[1:]) if chunk[0] in "+-" else ("+", chunk)
        coeff_match = re.match(r"\d+", body)
        coeff = int(coeff_match.group()) if coeff_match else 1
        rest = body[coeff_match.end():] if coeff_match else body
        exps = [0] * len(VARS)
        pos = 0
        for m in _TERM.finditer(rest):
            if m.start() != pos:
                raise ValueError(f"cannot parse term {body!r} in {s!r}")
            k = VARS.index(m.group(1))
            exps[k] += int(m.group(2)) if m.group(2) else 1
            dim = max(dim, k + 1)
            pos = m.end()
        if pos != len(rest):
            raise ValueError(f"cannot parse term {body!r} in {s!r}")
        e = tuple(exps)
        terms[e] = terms.get(e, 0) + (-coeff if sign == "-" else coeff)
    return {e[:dim]: c for e, c in terms.items()}, dim


# ====================================================================== group-ring arithmetic
def ring_mul(a, b, p=2):
    """Product in F_p[G] of two arrays of the same shape: a cyclic convolution.

    Every nonzero entry of the sparser factor is a monomial x^r; multiplying by it
    rolls the other factor by r (``np.roll`` wraps around: periodic boundaries).
    """
    if np.count_nonzero(a) > np.count_nonzero(b):
        a, b = b, a
    out = np.zeros_like(b)
    for r in zip(*np.nonzero(a)):
        out += a[r] * np.roll(b, r, axis=tuple(range(b.ndim)))
    return out % p


def ring_pow(a, n, p=2):
    result = np.zeros_like(a)
    result[(0,) * a.ndim] = 1
    while n:
        if n & 1:
            result = ring_mul(result, a, p)
        a = ring_mul(a, a, p)
        n >>= 1
    return result


def mult_matrix(f, shape, p=2):
    """Matrix of "multiply by f" on F_p[G], sites flattened row-major.

    Column k is f * delta_k = f translated to site k: the stencil stamped down
    once. ``f`` is a ``Poly`` or a group-ring array.
    """
    shape = tuple(shape)
    a = f.on(shape, p) if isinstance(f, Poly) else np.asarray(f) % p
    N = int(np.prod(shape))
    sites = np.indices(shape).reshape(len(shape), N)          # coordinates of every site
    M = np.zeros((N, N), dtype=np.uint8 if p == 2 else np.int64)
    for off in zip(*np.nonzero(a)):
        target = np.ravel_multi_index(
            tuple((sites[k] + off[k]) % shape[k] for k in range(len(shape))), shape)
        M[target, np.arange(N)] = (M[target, np.arange(N)] + a[off]) % p
    return M


# ====================================================================== dim R/I
def quotient_dim(gens, shape, p=2, eliminate=True):
    """dim_{F_p} R/I, for R = F_p[G] and I the ideal generated by ``gens``.

    Brute force: I is spanned by every translate of every generator, so
    dim I = rank of the (N x kN) matrix [M_g1 | M_g2 | ...], and dim R/I = N - dim I.

    With ``eliminate=True`` (default) a generator that is *linear* in some variable,
    g = c * v^(+-1) * m + h with h free of v (m a monomial), is used first to remove v:
    in R/I, v = H := (-h / (c m))^(+-1), so R/I = R'/(H^L - 1, other gens with v -> H),
    one dimension down. Haah's cubic code has g = 1 + x + y + z, which turns an
    L^3-site problem into an L^2-site one. Both routes give the same number;
    ``tests/`` checks it.
    """
    shape = tuple(shape)
    gens = [g.on(shape, p) if isinstance(g, Poly) else np.asarray(g) % p for g in gens]
    gens = [g for g in gens if g.any()]
    while eliminate and len(shape) > 1:
        step = _find_linear(gens, p)
        if step is None:
            break
        i, axis = step
        gens, shape = _eliminate(gens, i, axis, shape, p)
    N = int(np.prod(shape))
    if not gens:
        return N
    big = np.concatenate([mult_matrix(g, shape, p) for g in gens], axis=1)
    return N - rank(big, p)


def _find_linear(gens, p):
    for i, g in enumerate(gens):
        for axis in range(g.ndim):
            moved = np.moveaxis(g, axis, 0)
            used = [k for k in range(moved.shape[0]) if moved[k].any()]
            # v appears in exactly one monomial, with exponent +1 or -1 (mod L)
            if len(used) == 2 and 0 in used:
                k = used[1]
                if (k == 1 or k == moved.shape[0] - 1) and np.count_nonzero(moved[k]) == 1:
                    return i, axis
    return None


def _eliminate(gens, i, axis, shape, p):
    L = shape[axis]
    g = np.moveaxis(gens[i], axis, 0)
    k = 1 if g[1].any() else L - 1
    h, lead = g[0], g[k]                                # g = lead * v^k + h
    (m,) = [tuple(r) for r in np.argwhere(lead)]
    c = int(lead[m])
    # v^k = -h / (c * x^m)  in the quotient
    inv_mono = np.zeros_like(h)
    inv_mono[tuple((-mi) % s for mi, s in zip(m, h.shape))] = (-pow(c, p - 2, p)) % p
    H = ring_mul(h, inv_mono, p)
    if k != 1:                                          # v = (v^-1)^-1 = H^(L-1), since H^L = 1
        H = ring_pow(H, L - 1, p)
    powers = [None] * L                                 # H^0 .. H^(L-1)
    powers[0] = np.zeros_like(H)
    powers[0][(0,) * H.ndim] = 1
    for j in range(1, L):
        powers[j] = ring_mul(powers[j - 1], H, p)
    new = [(ring_mul(powers[-1], H, p) - powers[0]) % p]   # H^L - 1
    for j, other in enumerate(gens):
        if j == i:
            continue
        o = np.moveaxis(other, axis, 0)
        image = np.zeros_like(H)
        for e in range(L):
            if o[e].any():
                image = (image + ring_mul(o[e], powers[e], p)) % p
        new.append(image)
    new_shape = shape[:axis] + shape[axis + 1:]
    return [n for n in new if n.any()], new_shape
