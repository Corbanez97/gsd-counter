"""Closed forms for the degeneracy of two families of 2D classical stencils (p = 2).

Both start from the same fact: for a classical stencil f,

    N_c = dim ker(multiply by f) = dim R/(f),      R = F_2[x, y]/(x^Lx - 1, y^Ly - 1).

1. **Newman-Moore family**, f = y + g(x). Modulo f, y = g(x), so y disappears:
   R/(f) = F_2[x]/(x^Lx - 1, g(x)^Ly - 1). F_2[x] is a principal ideal domain, so

       N_c = deg gcd( x^Lx + 1 , g(x)^Ly + 1 ).

   Newman-Moore is g = 1 + x, the trinomial stencil is g = 1 + x + x^2.

2. **The corner stencil** ``haah_z``, f = 1 + x + x^2 + y + xy + y^2, is quadratic in y,
   so y cannot be eliminated over F_2. Over F_4 = {0, 1, w, w^2} it factors into two
   lines, f = (x + w y + w^2)(x + w^2 y + w), and counting their points on the torus gives

       N_c(Lx, Ly) = 2 min(2^a, 2^b) N(mx, my) - [a == b],
       N(mx, my)   = deg gcd_{F_4[y]}( y^my + 1 , (w y + w^2)^mx + 1 ),

   with Lx = 2^a mx, Ly = 2^b my (mx, my odd). Derivation in the README.
"""

from __future__ import annotations

__all__ = ["newman_moore_family", "haah_z", "curve_points", "split2"]


# ====================================================================== F_2[x] as int bitmasks
def _mul2(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def _mod2(a, b):
    db = b.bit_length()
    while a.bit_length() >= db:
        a ^= b << (a.bit_length() - db)
    return a


def _gcd2(a, b):
    while b:
        a, b = b, _mod2(a, b)
    return a


def _powmod2(base, e, mod):
    r, base = 1, _mod2(base, mod)
    while e:
        if e & 1:
            r = _mod2(_mul2(r, base), mod)
        base = _mod2(_mul2(base, base), mod)
        e >>= 1
    return r


def newman_moore_family(Lx, Ly, g_exponents=(0, 1)):
    """N_c for f = y + g(x), with g given by its exponents (default 1 + x: Newman-Moore)."""
    g = 0
    for e in g_exponents:
        g ^= 1 << e
    modulus = (1 << Lx) | 1                                 # x^Lx + 1
    h = _powmod2(g, Ly, modulus) ^ 1                        # g^Ly + 1 (mod x^Lx + 1)
    return _gcd2(modulus, h).bit_length() - 1 if h else Lx


# ====================================================================== F_4[y] as lists
# F_4 = {0, 1, w, w^2} encoded as {0, 1, 2, 3}
_LOG, _EXP = {1: 0, 2: 1, 3: 2}, {0: 1, 1: 2, 2: 3}
_MUL4 = [[0 if a == 0 or b == 0 else _EXP[(_LOG[a] + _LOG[b]) % 3] for b in range(4)]
         for a in range(4)]
_INV4 = {1: 1, 2: 3, 3: 2}


def _norm(p):
    while p and p[-1] == 0:
        p.pop()
    return p


def _mul4(a, b):
    if not a or not b:
        return []
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y:
                    r[i + j] ^= _MUL4[x][y]
    return _norm(r)


def _mod4(a, b):
    a, db, ib = a[:], len(b) - 1, _INV4[b[-1]]
    while a and len(a) - 1 >= db:
        c, sh = _MUL4[a[-1]][ib], len(a) - 1 - db
        for i, y in enumerate(b):
            if y:
                a[i + sh] ^= _MUL4[c][y]
        _norm(a)
    return a


def _gcd4(a, b):
    a, b = _norm(a[:]), _norm(b[:])
    while b:
        a, b = b, _mod4(a, b)
    return a


def _powmod4(base, e, mod):
    r, b = [1], _mod4(base, mod)
    while e:
        if e & 1:
            r = _mod4(_mul4(r, b), mod)
        b = _mod4(_mul4(b, b), mod)
        e >>= 1
    return r


def curve_points(mx, my):
    """N(mx, my): number of eta in mu_my with w*eta + w^2 in mu_mx (one line of the conic)."""
    mod = [1] + [0] * (my - 1) + [1]                        # y^my + 1
    h = _norm(_powmod4([3, 2], mx, mod)) or [0]             # (w^2 + w y)^mx
    h[0] ^= 1                                               # ... + 1
    _norm(h)
    return my if not h else len(_gcd4(mod, h)) - 1


def split2(L):
    """L = 2^a * m with m odd -> (a, m)."""
    a = 0
    while L % 2 == 0:
        L //= 2
        a += 1
    return a, L


def haah_z(Lx, Ly):
    """N_c of the corner stencil 1 + x + x^2 + y + xy + y^2 on an Lx x Ly torus."""
    a, mx = split2(Lx)
    b, my = split2(Ly)
    return 2 * min(2 ** a, 2 ** b) * curve_points(mx, my) - (1 if a == b else 0)
