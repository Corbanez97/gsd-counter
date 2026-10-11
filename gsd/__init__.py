"""gsd: ground-state degeneracy of translation-invariant stabilizer models over F_p.

    lattice -> group ring    ``Poly("1 + x + y").on((L, L))``
    stencil -> matrix         ``mult_matrix(f, shape)``
    model   -> degeneracy     ``MODELS["haah_cubic"].log_gsd(8)``

See README.md for the story and notebooks/ for the worked path through it.
"""

from . import closed_form, fractals
from .linalg import nullspace, rank, rref
from .models import HAAH_CUBIC, MODELS, NEWMAN_MOORE, STENCILS, TORIC_CODE, Generator, Model, classical
from .ring import Poly, mult_matrix, quotient_dim, ring_mul, ring_pow

__all__ = [
    "Poly", "ring_mul", "ring_pow", "mult_matrix", "quotient_dim",
    "rank", "rref", "nullspace",
    "Model", "Generator", "classical", "MODELS", "STENCILS",
    "TORIC_CODE", "NEWMAN_MOORE", "HAAH_CUBIC",
    "closed_form", "fractals",
]
