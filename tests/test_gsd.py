"""Every number quoted in the README is checked here.  Run: python -m pytest -q"""

import itertools

import numpy as np
import pytest

from gsd import (HAAH_CUBIC, MODELS, NEWMAN_MOORE, STENCILS, TORIC_CODE, Poly, classical,
                 closed_form, fractals, mult_matrix, nullspace, quotient_dim, rank, rref)


# ---------------------------------------------------------------- linear algebra
def slow_rank(A, p):
    """Textbook elimination, one row at a time: the reference for the fast routines."""
    M = np.array(A, dtype=np.int64) % p
    r = 0
    for c in range(M.shape[1]):
        rows = [i for i in range(r, M.shape[0]) if M[i, c]]
        if not rows:
            continue
        M[[r, rows[0]]] = M[[rows[0], r]]
        M[r] = M[r] * pow(int(M[r, c]), p - 2, p) % p
        for i in range(M.shape[0]):
            if i != r:
                M[i] = (M[i] - M[i, c] * M[r]) % p
        r += 1
    return r


@pytest.mark.parametrize("p", [2, 3, 5])
def test_rank_and_nullspace_random(p):
    rng = np.random.default_rng(p)
    for _ in range(40):
        rows, cols = rng.integers(1, 90, size=2)
        A = rng.integers(0, p, size=(rows, cols)) * (rng.random((rows, cols)) < 0.3)
        r = rank(A, p)
        assert r == slow_rank(A, p)
        K = nullspace(A, p)
        assert K.shape == (cols - r, cols)
        assert not ((A @ K.T) % p).any()
        assert rank(K, p) == K.shape[0]


def test_rref_hand_example():
    # the worked example of notebooks/01: R_2 = R_0 + R_1
    A = [[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]]
    R, piv = rref(A)
    assert piv == [0, 1]
    assert R.tolist() == [[1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]]
    assert sorted(map(tuple, nullspace(A).tolist())) == [(1, 0, 0, 1), (1, 1, 1, 0)]


# ---------------------------------------------------------------- group ring
def test_poly_parsing_roundtrip():
    for s in ["1 + x + y", "1 + x^-1 + y^-1", "1 + xy + yz + zx", "2x - y^2"]:
        assert Poly(str(Poly(s))) == Poly(s)
    assert Poly("1 + x + y").bar() == Poly("1 + x^-1 + y^-1")


def test_mult_matrix_is_rolling():
    f = STENCILS["haah_z"]
    shape = (5, 7)
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, size=shape)
    rolled = sum(np.roll(t, off, axis=(0, 1)) for off, _ in f.offsets()) % 2
    assert (mult_matrix(f, shape) @ t.reshape(-1) % 2 == rolled.reshape(-1)).all()


def test_frobenius():
    f = Poly("1 + x + y")
    assert f.pow(8) == f.substitute(x=8, y=8)


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("name", list(MODELS))
def test_algebra_matches_brute_force(name):
    m = MODELS[name]
    sizes = range(2, 7) if m.dim == 2 else range(2, 6)
    for L in sizes:
        assert m.commutes(L)
        assert m.log_gsd(L, method="algebra") == m.log_gsd(L, method="brute"), (name, L)


def test_rectangular_and_eliminate_flag():
    for Lx, Ly in itertools.product(range(2, 9), repeat=2):
        f = NEWMAN_MOORE.generators[0].z[0]
        brute = NEWMAN_MOORE.log_gsd((Lx, Ly), method="brute")
        assert quotient_dim([f], (Lx, Ly), eliminate=False) == brute
        assert quotient_dim([f], (Lx, Ly)) == brute
    assert HAAH_CUBIC.log_gsd((2, 3, 4), method="brute") == HAAH_CUBIC.log_gsd((2, 3, 4))


def test_toric_code_is_size_independent():
    for L in [(2, 2), (3, 5), (8, 8), (32, 32)]:
        assert TORIC_CODE.log_gsd(L) == 2
    for p in (3, 5):
        assert TORIC_CODE.commutes((3, 4), p)
        assert TORIC_CODE.log_gsd((3, 4), p, method="brute") == 2


def test_haah_cubic_powers_of_two():
    for a in range(1, 6):
        assert HAAH_CUBIC.log_gsd(2 ** a) == 4 * 2 ** a - 2


def test_poster_numbers():
    F = NEWMAN_MOORE.check_matrix(7)
    assert (rank(F), NEWMAN_MOORE.n_qudits(7)) == (43, 49)
    assert NEWMAN_MOORE.log_gsd(63) == 62
    assert NEWMAN_MOORE.log_gsd(64) == 0


# ---------------------------------------------------------------- closed forms
def test_newman_moore_family_closed_form():
    tri = classical("tri", STENCILS["trinomial"])
    for Lx, Ly in itertools.product(range(2, 17), repeat=2):
        assert closed_form.newman_moore_family(Lx, Ly) == NEWMAN_MOORE.log_gsd((Lx, Ly), method="brute")
        assert closed_form.newman_moore_family(Lx, Ly, (0, 1, 2)) == tri.log_gsd((Lx, Ly), method="brute")


def test_haah_z_closed_form():
    m = classical("haah_z", STENCILS["haah_z"])
    for Lx, Ly in itertools.product(range(2, 21), repeat=2):
        assert closed_form.haah_z(Lx, Ly) == m.log_gsd((Lx, Ly), method="brute"), (Lx, Ly)
    assert [closed_form.haah_z(L, L) for L in (15, 16, 63, 85, 91)] == [27, 31, 123, 49, 25]
    assert (closed_form.haah_z(256, 256), closed_form.haah_z(256, 255)) == (511, 2)


def test_transpose_has_same_degeneracy():
    z, x = classical("z", STENCILS["haah_z"]), classical("x", STENCILS["haah_x"])
    for L in range(2, 12):
        assert z.log_gsd(L) == x.log_gsd(L)


# ---------------------------------------------------------------- fractals
def test_open_fractal_operators():
    u, corners = fractals.power_pattern(Poly("1 + x^-1 + y^-1"), 5)
    assert (len(u), len(corners)) == (3 ** 5, 3)
    u, corners = fractals.power_pattern(Poly("1 + x^-1 + y^-1 + z^-1"), 5)
    assert (len(u), len(corners)) == (4 ** 5, 4)


def test_closed_symmetries_commute():
    F = NEWMAN_MOORE.check_matrix(15)
    S = NEWMAN_MOORE.symmetries(15)
    assert S.shape[0] == NEWMAN_MOORE.log_gsd(15)
    assert not ((F[:, 225:] @ S.T) % 2).any()
    T = NEWMAN_MOORE.relations(15)
    assert T.shape[0] == S.shape[0]
    assert not ((T @ F) % 2).any()
    assert TORIC_CODE.symmetries(4).shape[0] == 2 * 32 - rank(TORIC_CODE.check_matrix(4))
