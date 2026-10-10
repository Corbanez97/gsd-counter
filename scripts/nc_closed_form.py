#!/usr/bin/env python3
"""Closed form for N_c = dim ker A of the `haah_z` stencil on an Ly x Lx torus / GF(2).

    N_c(Lx, Ly) = 2 * min(2^a, 2^b) * N(mx, my)  -  [a == b]

with Lx = 2^a * mx, Ly = 2^b * my (mx, my odd) and

    N(mx, my) = deg gcd_{F4[y]}( y^my + 1 ,  (w*y + w^2)^mx + 1 ),   w^2 + w + 1 = 0.

Derivation and numerical validation: docs/haah_z_gsd_closed_form.md.

    python3 scripts/nc_closed_form.py 15 10        # single value
    python3 scripts/nc_closed_form.py --verify 40  # compare with nc_sweep brute force
"""

import argparse

# ---------------------------------------------------------------- F4 = {0, 1, w, w^2} as {0, 1, 2, 3}
_LOG = {1: 0, 2: 1, 3: 2}
_EXP = {0: 1, 1: 2, 2: 3}
MUL = [[0 if a == 0 or b == 0 else _EXP[(_LOG[a] + _LOG[b]) % 3]
        for b in range(4)] for a in range(4)]
INV = {1: 1, 2: 3, 3: 2}


def _norm(p):
    while p and p[-1] == 0:
        p.pop()
    return p


def _mul(a, b):
    if not a or not b:
        return []
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            row = MUL[x]
            for j, y in enumerate(b):
                if y:
                    r[i + j] ^= row[y]
    return _norm(r)


def _mod(a, b):
    a, db, ib = a[:], len(b) - 1, INV[b[-1]]
    while a and len(a) - 1 >= db:
        c, sh = MUL[a[-1]][ib], len(a) - 1 - db
        for i, y in enumerate(b):
            if y:
                a[i + sh] ^= MUL[c][y]
        _norm(a)
    return a


def _gcd(a, b):
    a, b = _norm(a[:]), _norm(b[:])
    while b:
        a, b = b, _mod(a, b)
    return a


def _powmod(base, e, mod):
    r, b = [1], _mod(base, mod)
    while e:
        if e & 1:
            r = _mod(_mul(r, b), mod)
        b = _mod(_mul(b, b), mod)
        e >>= 1
    return r


def curve_points(mx, my):
    """N(mx, my): # of eta in mu_my with w*eta + w^2 in mu_mx (one branch of the conic)."""
    mod = [1] + [0] * (my - 1) + [1]            # y^my + 1
    h = _norm(_powmod([3, 2], mx, mod))         # (w^2 + w*y)^mx  mod (y^my + 1)
    h = h or [0]
    h[0] ^= 1                                   # ... + 1
    _norm(h)
    return my if not h else len(_gcd(mod, h)) - 1


def split2(L):
    a = 0
    while L % 2 == 0:
        L //= 2
        a += 1
    return a, L


def Nc_closed(Lx, Ly):
    a, mx = split2(Lx)
    b, my = split2(Ly)
    return 2 * min(2 ** a, 2 ** b) * curve_points(mx, my) - (1 if a == b else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("Lx", type=int, nargs="?")
    ap.add_argument("Ly", type=int, nargs="?")
    ap.add_argument("--verify", type=int, metavar="LMAX",
                    help="check against brute-force nc_sweep for all 2 <= Lx,Ly <= LMAX")
    args = ap.parse_args()

    if args.verify:
        from nc_sweep import Nc, OPS
        bad = [(lx, ly, Nc(OPS["haah_z"], lx, ly, 2), Nc_closed(lx, ly))
               for lx in range(2, args.verify + 1) for ly in range(2, args.verify + 1)
               if Nc(OPS["haah_z"], lx, ly, 2) != Nc_closed(lx, ly)]
        n = (args.verify - 1) ** 2
        print(f"{n - len(bad)}/{n} pairs agree" + (f"; mismatches: {bad[:10]}" if bad else ""))
        return

    if args.Lx is None:
        ap.error("give Lx (and Ly), or --verify LMAX")
    Ly = args.Ly if args.Ly is not None else args.Lx
    n = Nc_closed(args.Lx, Ly)
    print(f"Lx={args.Lx} Ly={Ly}  Nc={n}  GSD=2^{n}")


if __name__ == "__main__":
    main()
