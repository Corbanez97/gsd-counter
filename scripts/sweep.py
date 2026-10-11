#!/usr/bin/env python3
"""Sweep log_p GSD over torus sizes and write a CSV (columns: Lx, Ly[, Lz], log_gsd).

Models: the stabilizer models (toric_code, newman_moore, haah_cubic) and every
classical stencil in gsd.STENCILS (haah_z, trinomial, six, ...).

    python scripts/sweep.py haah_cubic --L 2-32                # cubic tori
    python scripts/sweep.py newman_moore --lx 2-64 --ly 2-64   # full rectangular grid
    python scripts/sweep.py haah_z --L 2-128 --method closed   # closed form, instant
    python scripts/sweep.py six --L 2-16 -p 3                  # over GF(3)
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gsd import MODELS, STENCILS, classical, closed_form  # noqa: E402


def get_model(name):
    if name in MODELS:
        return MODELS[name]
    if name in STENCILS:
        return classical(name, STENCILS[name])
    raise SystemExit(f"unknown model {name!r}; choose from {sorted(set(MODELS) | set(STENCILS))}")


CLOSED = {
    "haah_z": closed_form.haah_z,
    "newman_moore": closed_form.newman_moore_family,
    "trinomial": lambda lx, ly: closed_form.newman_moore_family(lx, ly, (0, 1, 2)),
}


def parse_range(spec):
    """'2-23' -> [2..23];  '8' -> [8];  '2,4,8' -> [2, 4, 8]."""
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--L", help="cubic/square tori, side lengths e.g. 2-32")
    ap.add_argument("--lx", help="Lx values (2D rectangular grid with --ly)")
    ap.add_argument("--ly", help="Ly values")
    ap.add_argument("-p", type=int, default=2, help="prime modulus (default 2)")
    ap.add_argument("--method", default="auto", choices=["auto", "brute", "algebra", "closed"])
    ap.add_argument("-o", "--out", help="CSV path (default data/<model>.csv)")
    args = ap.parse_args()

    model = get_model(args.model)
    if args.L:
        sizes = [(L,) * model.dim for L in parse_range(args.L)]
    elif args.lx and args.ly and model.dim == 2:
        sizes = [(lx, ly) for ly in parse_range(args.ly) for lx in parse_range(args.lx)]
    else:
        ap.error("give --L, or --lx and --ly for a 2D model")

    if args.method == "closed":
        if args.model not in CLOSED or args.p != 2:
            ap.error(f"closed forms exist for {sorted(CLOSED)} at p = 2")
        compute = lambda s: CLOSED[args.model](*s)                        # noqa: E731
    else:
        compute = lambda s: model.log_gsd(s, args.p, args.method)        # noqa: E731

    path = args.out or os.path.join("data", f"{args.model}.csv")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Lx", "Ly", "Lz"][: model.dim] + ["log_gsd"])
        for s in sizes:
            k = compute(s)
            w.writerow(list(s) + [k])
            fh.flush()
            print(f"{args.model} p={args.p}  L={'x'.join(map(str, s)):>10}  log_{args.p} GSD = {k}", flush=True)
    print(f"\nwrote {len(sizes)} rows to {path}")


if __name__ == "__main__":
    main()
