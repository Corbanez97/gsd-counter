"""Animated Gaussian elimination (RREF) and kernel extraction over GF(p).

The scene runs the full pipeline of the gsd package:
build A from a *parametrised* operator, reduce it to RREF, read off the rank,
and then **construct each nullspace basis vector on screen** — one free column
at a time, back-substituting v[pivot] = -R[i][free] into a slot row that sits
directly underneath the matrix columns it comes from.

Configure the operator and the torus at the top of this file, or per render
from the shell::

    OP=nm LX=3 LY=3 P=2 venv/bin/manim -ql scripts/animate_rref.py OperatorKernel
    OP=six LX=4 LY=4 P=2 venv/bin/manim -qh scripts/animate_rref.py OperatorKernel

Ready-made scenes: ``OperatorKernel`` (the configurable one), ``SixNeighborL4``,
``NewmanMooreL3``, ``NewmanMooreL3GF3``, ``TwoRollL3x2``, plus the toy scenes
``VisualGFpRREF``, ``KernelGF2Ring`` and ``KernelGF2TwoFree``.

Pacing knobs live on the scene class: ``max_detailed_vectors`` (how many basis
vectors get the slow step-by-step back-substitution), ``max_shown_vectors``
(how many are built at all) and ``max_montage_vectors`` (how many appear in the
closing frame).

No LaTeX required: every label is a Pango ``Text`` and the matrix brackets are
drawn as plain polylines, so this renders on a machine with no TeX install.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from manim import *


# ---------------------------------------------------------------------------
# Operators come from the gsd package, so the animation cannot drift away from
# the code it illustrates. A stencil is a polynomial; its matrix is "multiply
# by f" on the torus (column k = the stencil stamped at site k).
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gsd import STENCILS, Poly, mult_matrix, nullspace, ring_mul  # noqa: E402

OPS = dict(STENCILS, plus4=Poly("x + x^-1 + y + y^-1"), square4=Poly("1 + x + y + xy"),
           nm_w2=Poly("2 + x + y"), nm=STENCILS["newman_moore"])


def stencil_op(f, shape):
    """The operator as a callable on (Lx, Ly) patterns: t -> f * t (rolls, mod p)."""
    return lambda t, p=2: ring_mul(f.on(shape, p), np.asarray(t) % p, p)


# ---------------------------------------------------------------------------
# Configuration. Edit here, or override any of them from the environment.
# ---------------------------------------------------------------------------

OP_NAME = os.environ.get("OP", "nm")
LX = int(os.environ.get("LX", 3))
LY = int(os.environ.get("LY", 3))
P = int(os.environ.get("P", 2))


# ---------------------------------------------------------------------------
# GF(p) helpers
# ---------------------------------------------------------------------------


def gfp_inv(a: int, p: int) -> int:
    a_mod = int(a) % p
    if a_mod == 0:
        raise ZeroDivisionError(f"0 has no inverse mod {p}")
    return pow(a_mod, p - 2, p)


class GFMatrix(VGroup):
    """A matrix of integers laid out on a fixed grid, with drawn brackets.

    Manim's built-in ``Matrix`` family renders its entries *and* its brackets
    through ``MathTex``, which needs a LaTeX toolchain. This builds the same
    picture out of ``Text`` glyphs on a grid we control, which also gives us
    stable per-cell handles to animate.
    """

    def __init__(
        self,
        values,
        cell_width: float = 0.85,
        cell_height: float = 0.8,
        font_size: int = 40,
        max_width: float | None = None,
        max_height: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.values = np.array(values, dtype=int)
        self.n_rows, self.n_cols = self.values.shape

        # Shrink the *grid parameters* rather than calling ``.scale()`` on the
        # finished mobject: every later ``Transform`` builds a fresh ``Text`` at
        # ``self.font_size``, so a post-hoc scale would make edited cells jump
        # back to full size.
        fit = 1.0
        if max_width is not None:
            fit = min(fit, max_width / (self.n_cols * cell_width))
        if max_height is not None:
            fit = min(fit, max_height / (self.n_rows * cell_height))

        self.cell_width = cell_width * fit
        self.cell_height = cell_height * fit
        self.font_size = max(8, int(round(font_size * fit)))

        # ``cells[i][j]`` is the live mobject for entry (i, j). Row swaps
        # permute this list so indices keep matching what is on screen.
        self.cells = [
            [self._make_entry(self.values[i, j], i, j) for j in range(self.n_cols)]
            for i in range(self.n_rows)
        ]
        self.entries = VGroup(*[c for row in self.cells for c in row])
        self.brackets = VGroup(self._make_bracket(LEFT), self._make_bracket(RIGHT))
        self.add(self.brackets, self.entries)

    # -- construction helpers -------------------------------------------------

    def _cell_position(self, i: int, j: int) -> np.ndarray:
        x = (j - (self.n_cols - 1) / 2) * self.cell_width
        y = ((self.n_rows - 1) / 2 - i) * self.cell_height
        return np.array([x, y, 0.0])

    def _make_entry(self, value: int, i: int, j: int) -> Text:
        return Text(str(int(value)), font_size=self.font_size).move_to(
            self._cell_position(i, j)
        )

    def _make_bracket(self, direction: np.ndarray) -> VMobject:
        half_w = self.n_cols * self.cell_width / 2
        half_h = self.n_rows * self.cell_height / 2
        # Offsets ride on the cell size so a shrunken grid keeps its proportions.
        x = direction[0] * (half_w + 0.12 * self.cell_width)
        tip = -direction[0] * 0.21 * self.cell_width  # serifs point back inward
        bracket = VMobject()
        bracket.set_points_as_corners(
            np.array(
                [
                    [x + tip, half_h, 0.0],
                    [x, half_h, 0.0],
                    [x, -half_h, 0.0],
                    [x + tip, -half_h, 0.0],
                ]
            )
        )
        bracket.set_stroke(color=WHITE, width=4)
        return bracket

    # -- accessors ------------------------------------------------------------

    def entry(self, i: int, j: int) -> Text:
        return self.cells[i][j]

    def row(self, i: int) -> VGroup:
        return VGroup(*self.cells[i])

    def column(self, j: int) -> VGroup:
        return VGroup(*[self.cells[i][j] for i in range(self.n_rows)])

    # -- animation builders ---------------------------------------------------

    def swap_rows_anims(self, i: int, k: int) -> list:
        """Slide row ``i`` and row ``k`` past each other, then re-index."""
        anims = []
        for j in range(self.n_cols):
            a, b = self.cells[i][j], self.cells[k][j]
            anims.append(a.animate.move_to(b.get_center()))
            anims.append(b.animate.move_to(a.get_center()))
        self.cells[i], self.cells[k] = self.cells[k], self.cells[i]
        self.values[[i, k]] = self.values[[k, i]]
        return anims

    def set_entry_anims(self, i: int, j: int, value, color=WHITE, force: bool = False) -> list:
        """Morph a single cell. Positions come from the live mobject, not from
        ``_cell_position``, so this keeps working after the matrix is moved."""
        value = int(value)
        if not force and value == int(self.values[i, j]):
            return []
        cell = self.cells[i][j]
        target = Text(str(value), font_size=self.font_size, color=color).move_to(
            cell.get_center()
        )
        self.values[i, j] = value
        return [Transform(cell, target)]

    def set_row_anims(self, i: int, new_row: np.ndarray, color=WHITE, force: bool = False) -> list:
        """Morph only the cells of row ``i`` whose value actually changed."""
        anims = []
        for j in range(self.n_cols):
            anims += self.set_entry_anims(i, j, new_row[j], color=color, force=force)
        return anims


class VisualGFpRREF(Scene):
    """Generic GF(p) demo. Subclass and override the attributes to retarget.

    The matrix is deliberately rank-deficient: a full-rank matrix has a trivial
    kernel, which makes the whole point of the epilogue vanish.
    """

    prime = 3
    # rows 2 and 3 are dependent: R2 = R0 + R1, R3 = 2*R0  ->  rank 2, N_c = 2.
    initial_array = np.array(
        [
            [1, 1, 0, 2],
            [0, 2, 1, 1],
            [1, 0, 1, 0],
            [2, 2, 0, 1],
        ],
        dtype=int,
    )
    title_text: str | None = None  # None -> "Gaussian Elimination over GF(p)"

    # The matrix never grows past this box, so the panel to its right always has
    # room and the kernel slot row has somewhere to sit underneath. The height
    # budget is tight: status banner above, slot row and its label below.
    matrix_max_width = 5.6
    matrix_max_height = 3.6

    # Set to (rows, cols) when a kernel vector is a folded-up lattice
    # configuration, so the epilogue draws it as a grid instead of a flat tuple.
    kernel_grid_shape: tuple[int, int] | None = None

    # The operator the matrix came from, if any: kernel vectors are verified
    # against *it*, not only against A, which closes the loop back to the roll
    # code the way the notebook's ``validate`` does.
    op = None

    # One animation per eliminated row is the clearest read, but a 16x16 with
    # fifteen rows to clear per pivot turns that into minutes of near-identical
    # clips. Batching clears the whole column in a single beat instead.
    batch_row_ops = False

    # Kernel phase: how many basis vectors get the slow back-substitution
    # treatment, and how many get built at all before the montage.
    max_detailed_vectors = 2
    max_shown_vectors = 4
    max_montage_vectors = 12

    # Right-hand column used by every panel once the matrix has slid left.
    panel_x = 3.0
    panel_max_width = 7.2

    def construct(self):
        prime = self.prime
        working_matrix = np.array(self.initial_array, dtype=int) % prime

        # -- header ----------------------------------------------------------
        title = Text(
            self.title_text or f"Gaussian Elimination over GF({prime})",
            font_size=40,
            weight=BOLD,
        ).to_edge(UP, buff=0.4)
        if title.width > config.frame_width - 1.0:
            title.scale_to_fit_width(config.frame_width - 1.0)
        underline = Underline(title, color=BLUE_B)
        self.add(title, underline)
        self.header = VGroup(title, underline)

        # ``status`` keeps a stable identity for the whole scene; every update
        # is a Transform into a freshly built Text placed in the same slot.
        self.status_anchor = underline
        self.status = self._make_status("Starting matrix reduction", YELLOW)
        self.add(self.status)

        # -- matrix ----------------------------------------------------------
        matrix = GFMatrix(
            working_matrix,
            max_width=self.matrix_max_width,
            max_height=self.matrix_max_height,
        )
        matrix.move_to(DOWN * 0.4)
        self.play(FadeIn(matrix.entries, shift=UP * 0.2), Create(matrix.brackets))
        self.wait(0.5)

        rows, cols = working_matrix.shape
        pivot_row = 0
        pivot_cols: list[int] = []

        for col in range(cols):
            if pivot_row >= rows:
                break

            self.set_status(f"Examining column {col}", BLUE)

            # 1. Find a pivot at or below the current row.
            found = None
            for r in range(pivot_row, rows):
                if working_matrix[r, col] != 0:
                    found = r
                    break

            if found is None:
                self.set_status(f"Column {col} has no pivot — it stays free", GREY_B)
                self.wait(0.4)
                continue

            highlight = SurroundingRectangle(
                matrix.entry(found, col), color=GREEN, buff=0.12
            )
            self.play(Create(highlight))

            # 2. Swap the pivot into position.
            if found != pivot_row:
                self.set_status(f"Swap row {pivot_row} and row {found}", ORANGE)
                # Grab the destination before the swap re-indexes the cells.
                destination = matrix.entry(pivot_row, col).get_center()
                swap_anims = matrix.swap_rows_anims(pivot_row, found)
                working_matrix[[pivot_row, found]] = working_matrix[[found, pivot_row]]
                self.play(*swap_anims, highlight.animate.move_to(destination))

            # 3. Normalise the pivot to 1.
            pivot_value = int(working_matrix[pivot_row, col])
            if pivot_value != 1:
                inv = gfp_inv(pivot_value, prime)
                self.set_status(
                    f"Scale row {pivot_row} by {inv}  "
                    f"({inv} is the inverse of {pivot_value} mod {prime})",
                    YELLOW,
                )
                working_matrix[pivot_row] = (working_matrix[pivot_row] * inv) % prime
                anims = matrix.set_row_anims(pivot_row, working_matrix[pivot_row])
                if anims:
                    self.play(*anims)

            # 4. Clear the rest of the column (full RREF, above and below).
            dirty = [
                r for r in range(rows) if r != pivot_row and working_matrix[r, col] != 0
            ]
            if self.batch_row_ops:
                if dirty:
                    self.set_status(
                        f"Clear column {col}: {len(dirty)} rows  -=  "
                        f"(their entry) x Row {pivot_row}   mod {prime}",
                        RED,
                    )
                    boxes = VGroup(
                        *[
                            SurroundingRectangle(matrix.row(r), color=RED, buff=0.06)
                            for r in dirty
                        ]
                    )
                    self.play(Create(boxes), run_time=0.5)
                    anims = []
                    for r in dirty:
                        factor = int(working_matrix[r, col])
                        working_matrix[r] = (
                            working_matrix[r] - factor * working_matrix[pivot_row]
                        ) % prime
                        anims += matrix.set_row_anims(r, working_matrix[r])
                    if anims:
                        self.play(*anims, run_time=1.2)
                    self.play(FadeOut(boxes), run_time=0.4)
            else:
                for r in dirty:
                    factor = int(working_matrix[r, col])
                    self.set_status(
                        f"Row {r}  =  (Row {r} - {factor} x Row {pivot_row})  mod {prime}",
                        RED,
                    )
                    target_row = SurroundingRectangle(matrix.row(r), color=RED, buff=0.12)
                    self.play(Create(target_row), run_time=0.4)

                    working_matrix[r] = (
                        working_matrix[r] - factor * working_matrix[pivot_row]
                    ) % prime
                    anims = matrix.set_row_anims(r, working_matrix[r])
                    if anims:
                        self.play(*anims)
                    self.play(FadeOut(target_row), run_time=0.3)

            self.play(FadeOut(highlight))
            pivot_cols.append(col)
            pivot_row += 1

        self.set_status("RREF complete — now read off the invariants", GREEN)
        self.play(Circumscribe(matrix, color=GREEN, run_time=1.5))
        self.show_invariants(matrix, working_matrix, pivot_cols, prime)
        self.show_kernel_vectors(matrix, working_matrix, pivot_cols, prime)

    # -- text helpers ---------------------------------------------------------

    def _make_status(self, message: str, color) -> Text:
        text = Text(message, font_size=26, color=color)
        max_width = config.frame_width - 1.5
        if text.width > max_width:
            text.scale_to_fit_width(max_width)
        return text.next_to(self.status_anchor, DOWN, buff=0.35)

    def set_status(self, message: str, color) -> None:
        self.play(Transform(self.status, self._make_status(message, color)),
                  run_time=0.6)

    def _fit(self, mobject, max_width: float | None = None):
        max_width = self.panel_max_width if max_width is None else max_width
        if mobject.width > max_width:
            mobject.scale_to_fit_width(max_width)
        return mobject

    @staticmethod
    def _power(base, exponent, color=WHITE, font_size: int = 32) -> VGroup:
        """``base^exponent`` without LaTeX: a smaller glyph raised to the top."""
        b = Text(str(base), font_size=font_size, color=color)
        e = Text(str(exponent), font_size=int(font_size * 0.62), color=color)
        e.next_to(b, RIGHT, buff=0.05).align_to(b, UP).shift(UP * 0.14)
        return VGroup(b, e)

    @staticmethod
    def _line(*parts, buff: float = 0.24) -> VGroup:
        """Lay mixed Text/_power pieces out on a shared baseline.

        A raised exponent widens its group's bounding box, so the buff has to
        be generous or the following ``=`` crowds it.
        """
        return VGroup(*parts).arrange(RIGHT, buff=buff, aligned_edge=DOWN)

    # -- epilogue 1: rank, nullity, dim Ker, GSD ------------------------------

    def show_invariants(self, matrix, rref, pivot_cols, prime) -> None:
        cols = matrix.n_cols
        free_cols = [c for c in range(cols) if c not in pivot_cols]
        rank, nullity = len(pivot_cols), len(free_cols)

        # Slide the matrix aside so the reasoning has somewhere to live, and up
        # so the kernel slot row fits underneath it later.
        x = -4.1
        left_limit = -config.frame_width / 2 + 0.35
        x += max(0.0, left_limit - (x - matrix.width / 2))
        self.play(matrix.animate.move_to(RIGHT * x + UP * 0.3))

        # 1. Pivot columns vs free columns, read straight off the RREF.
        self.set_status("Pivot columns set the rank; free columns set the kernel", WHITE)
        self.column_boxes = VGroup()
        for c in range(cols):
            color = GREEN if c in pivot_cols else YELLOW
            self.column_boxes.add(
                SurroundingRectangle(matrix.column(c), color=color, buff=0.07)
            )
        lag = 0.3 if cols <= 8 else 0.12
        self.play(LaggedStart(*[Create(b) for b in self.column_boxes], lag_ratio=lag))

        legend = VGroup(
            Text("pivot column", font_size=19, color=GREEN),
            Text("free column", font_size=19, color=YELLOW),
        ).arrange(RIGHT, buff=0.5)
        legend.next_to(self.column_boxes, DOWN, buff=0.15)
        self._fit(legend, matrix.width + 1.0)
        self.play(FadeIn(legend, shift=UP * 0.1))
        self.legend = legend

        # 2. The counting argument, one line at a time.
        fs = 25
        lines = [
            self._line(Text(f"columns of A:   n = {cols}", font_size=fs)),
            self._line(Text(f"rank A  =  # pivot columns  =  {rank}",
                            font_size=fs, color=GREEN)),
            self._line(Text(f"nullity  =  n - rank  =  {cols} - {rank}  =  {nullity}",
                            font_size=fs, color=YELLOW)),
            self._line(Text(f"N_c  =  dim Ker(A)  =  {nullity}",
                            font_size=fs, color=BLUE_B)),
        ]
        panel = VGroup(*lines).arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        panel.move_to(RIGHT * self.panel_x + UP * 1.25)
        self._fit(panel)

        for line in lines:
            self.play(FadeIn(line, shift=RIGHT * 0.2), run_time=0.55)

        # 3. Ground state degeneracy: every kernel vector is one ground state.
        factor = "doubles" if prime == 2 else f"multiplies by {prime}"
        if nullity == 0:
            message = "Kernel is trivial — no free directions, GSD = 1"
        elif nullity == 1:
            message = f"The single free direction {factor} the degeneracy"
        else:
            message = f"Each of the {nullity} free directions {factor} the degeneracy"
        self.set_status(message, GOLD)
        eq = lambda: Text("=", font_size=34, color=GOLD)
        gsd = self._line(
            Text("GSD", font_size=34, color=GOLD),
            eq(),
            self._power(prime, "N_c", color=GOLD, font_size=34),
            eq(),
            self._power(prime, nullity, color=GOLD, font_size=34),
            eq(),
            Text(str(prime ** nullity), font_size=34, color=GOLD),
        )
        gsd.next_to(panel, DOWN, buff=0.4).align_to(panel, LEFT)
        self._fit(gsd)
        box = SurroundingRectangle(gsd, color=GOLD, buff=0.2, corner_radius=0.1)
        self.play(Write(gsd), run_time=1.2)
        self.play(Create(box))
        self.wait(1.0)

        self.invariants_panel = VGroup(panel, gsd, box)

    # -- epilogue 2: the kernel vectors themselves ----------------------------

    def _verify(self, v: np.ndarray, prime: int) -> bool:
        """A v = 0 over GF(p), checked against the ORIGINAL matrix — and, when
        the operator is known, against the roll code itself."""
        ok = bool(((np.array(self.initial_array, dtype=int) @ v) % prime == 0).all())
        if self.op is not None and self.kernel_grid_shape is not None:
            rows, cols = self.kernel_grid_shape
            ok &= bool((self.op(v.reshape(rows, cols), prime) == 0).all())
        return ok

    def _fold(self, v: np.ndarray, max_width: float = 2.4) -> GFMatrix:
        """A kernel vector as the lattice configuration it actually is."""
        rows, cols = self.kernel_grid_shape
        return GFMatrix(
            v.reshape(rows, cols),
            cell_width=0.5,
            cell_height=0.48,
            font_size=26,
            max_width=max_width,
            max_height=2.4,
        )

    def set_detail(self, message: str, color=WHITE) -> None:
        """A second banner, in the right-hand panel, for the current step."""
        text = Text(message, font_size=22, color=color)
        self._fit(text)
        text.move_to(RIGHT * self.panel_x + UP * 1.75)
        if getattr(self, "detail", None) is None:
            self.detail = text
            self.play(FadeIn(self.detail), run_time=0.4)
        else:
            self.play(Transform(self.detail, text), run_time=0.5)

    def show_kernel_vectors(self, matrix, rref, pivot_cols, prime) -> None:
        n = matrix.n_cols
        free_cols = [c for c in range(n) if c not in pivot_cols]
        basis = nullspace(self.initial_array, prime)  # = the RREF read-off above

        self.play(FadeOut(self.invariants_panel))
        if not free_cols:
            self.set_status("No free columns — the kernel is trivial, GSD = 1", GREY_B)
            self.wait(2.0)
            return

        self.set_status(
            "Now build the kernel vectors: one per free column, by back-substitution",
            BLUE_B,
        )

        # The slot row lives directly under the matrix and shares its column
        # pitch, so component j of v sits under column j of A. That alignment is
        # the whole point: v[pivot] is read off the free column above it.
        slots = GFMatrix(
            np.zeros((1, n), dtype=int),
            cell_width=matrix.cell_width,
            cell_height=matrix.cell_height,
            font_size=matrix.font_size,
        )
        slots.next_to(matrix, DOWN, buff=0.5)
        slots.match_x(matrix)
        slot_label = Text("v =", font_size=max(14, matrix.font_size)).next_to(
            slots, LEFT, buff=0.15
        )
        self.play(
            self.legend.animate.next_to(slots, DOWN, buff=0.3),
            FadeIn(slots),
            FadeIn(slot_label),
        )
        self.detail = None
        self.set_detail("v is indexed by the columns of A, so it lines up underneath")

        for k, f in enumerate(free_cols[: self.max_shown_vectors]):
            detailed = k < self.max_detailed_vectors
            v = basis[free_cols.index(f)]

            # Reset the slots (force=True so colours from the last vector go).
            reset = slots.set_row_anims(0, np.zeros(n, dtype=int), force=True)
            self.play(*reset, run_time=0.4)

            self.set_status(f"Kernel vector v{k}  from free column {f}", YELLOW)
            col_box = SurroundingRectangle(matrix.column(f), color=YELLOW, buff=0.06)
            slot_box = SurroundingRectangle(slots.entry(0, f), color=YELLOW, buff=0.06)
            self.play(Create(col_box), Create(slot_box), run_time=0.5)

            self.set_detail(
                f"free variables: v[{f}] = 1, every other free entry = 0", YELLOW
            )
            self.play(*slots.set_entry_anims(0, f, 1, color=YELLOW), run_time=0.5)

            # Back-substitution. Row i of the RREF says
            #     v[pivot_i] + sum_over_free_g R[i, g] v[g] = 0,
            # and only v[f] is nonzero among the free entries, so
            #     v[pivot_i] = -R[i, f].
            for i, pc in enumerate(pivot_cols):
                coeff = int(rref[i, f]) % prime
                if coeff == 0:
                    continue
                value = (-coeff) % prime
                source = matrix.entry(i, f)

                if detailed:
                    self.set_detail(
                        f"row {i}:  v[{pc}] + {coeff}·v[{f}] = 0    "
                        f"->   v[{pc}] = -{coeff} = {value}  (mod {prime})",
                        GREEN,
                    )
                    pivot_box = SurroundingRectangle(
                        matrix.entry(i, pc), color=GREEN, buff=0.05
                    )
                    source_box = SurroundingRectangle(source, color=RED, buff=0.05)
                    self.play(Create(pivot_box), Create(source_box), run_time=0.4)

                    # Carry the coefficient down to its slot, then let the slot
                    # settle on -coeff: for p = 2 the digit is unchanged, for
                    # p > 2 the sign flip happens in front of you.
                    travelling = source.copy()
                    self.add(travelling)
                    self.play(
                        travelling.animate.move_to(
                            slots.entry(0, pc).get_center()
                        ).set_color(GREEN),
                        run_time=0.7,
                    )
                    self.play(
                        *slots.set_entry_anims(0, pc, value, color=GREEN, force=True),
                        FadeOut(travelling),
                        run_time=0.4,
                    )
                    self.play(FadeOut(pivot_box), FadeOut(source_box), run_time=0.3)
                else:
                    self.play(
                        *slots.set_entry_anims(0, pc, value, color=GREEN, force=True),
                        run_time=0.25,
                    )

            if not detailed:
                self.set_detail(
                    f"same recipe, played straight through:  "
                    f"v[pivot i] = -R[i][{f}]  (mod {prime})",
                    GREEN,
                )

            # Fold the vector back into the lattice and check it against the
            # operator it came from.
            ok = self._verify(v, prime)
            check = Text(
                ("A v = 0  and  op(v) = 0  [ok]" if self.op is not None
                 else "A v = 0  [ok]") if ok else "VERIFICATION FAILED",
                font_size=21,
                color=GREEN if ok else RED,
            )
            if self.kernel_grid_shape is not None:
                folded = self._fold(v)
                caption = Text(
                    f"v{k} as a {self.kernel_grid_shape[0]}x{self.kernel_grid_shape[1]} "
                    "configuration",
                    font_size=20,
                    color=YELLOW,
                )
                shown = VGroup(caption, folded, check).arrange(DOWN, buff=0.22)
            else:
                flat = Text(
                    f"v{k} = ({', '.join(str(int(x)) for x in v)})",
                    font_size=21,
                    color=YELLOW,
                )
                shown = VGroup(flat, check).arrange(DOWN, buff=0.22)
            shown.move_to(RIGHT * self.panel_x + DOWN * 0.75)
            self._fit(shown)
            self.play(FadeIn(shown, shift=UP * 0.15), run_time=0.7)
            self.wait(1.2 if detailed else 0.6)

            self.play(
                FadeOut(col_box), FadeOut(slot_box),
                FadeOut(shown), run_time=0.4,
            )

        remaining = len(free_cols) - min(len(free_cols), self.max_shown_vectors)
        if remaining:
            self.set_detail(
                f"the other {remaining} free columns give {remaining} more vectors "
                "the same way",
                GREY_B,
            )
            self.wait(1.0)

        self.show_basis_montage(basis, prime)

    def show_basis_montage(self, basis: np.ndarray, prime: int) -> None:
        """Closing frame: the whole basis at once, with the GSD it implies."""
        n_vectors = len(basis)
        verified = all(self._verify(v, prime) for v in basis)

        keep = {self.status, *self.header}
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep])
        self.set_status(
            f"The kernel: {n_vectors} independent configurations, "
            f"GSD = {prime}^{n_vectors} = {prime ** n_vectors}",
            GOLD,
        )

        shown = basis[: self.max_montage_vectors]
        if self.kernel_grid_shape is not None:
            tiles = VGroup()
            for i, v in enumerate(shown):
                grid = self._fold(v, max_width=1.9)
                label = Text(f"v{i}", font_size=18, color=YELLOW)
                tiles.add(VGroup(grid, label.next_to(grid, DOWN, buff=0.1)))
            per_row = min(4, max(1, len(tiles)))
            rows = VGroup(
                *[
                    VGroup(*tiles[i : i + per_row]).arrange(RIGHT, buff=0.45, aligned_edge=UP)
                    for i in range(0, len(tiles), per_row)
                ]
            ).arrange(DOWN, buff=0.5)
        else:
            rows = VGroup(
                *[
                    Text(
                        f"v{i} = ({', '.join(str(int(x)) for x in v)})",
                        font_size=22,
                        color=YELLOW,
                    )
                    for i, v in enumerate(shown)
                ]
            ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)

        pieces = [rows]
        if n_vectors > len(shown):
            pieces.append(
                Text(f"... and {n_vectors - len(shown)} more", font_size=20, color=GREY_B)
            )
        pieces.append(
            Text(
                "every one verified: A v = 0 over GF(%d)" % prime
                if verified
                else "VERIFICATION FAILED",
                font_size=22,
                color=GREEN if verified else RED,
            )
        )
        montage = VGroup(*pieces).arrange(DOWN, buff=0.35)
        montage.move_to(DOWN * 0.6)
        if montage.width > config.frame_width - 1.2:
            montage.scale_to_fit_width(config.frame_width - 1.2)
        if montage.height > 5.4:
            montage.scale_to_fit_height(5.4)
        self.play(
            LaggedStart(*[FadeIn(p_, shift=UP * 0.15) for p_ in montage], lag_ratio=0.25)
        )
        self.wait(3)


# ---------------------------------------------------------------------------
# Operator scenes. ``initial_array`` has to exist at class-creation time, so the
# scenes are stamped out by a small factory rather than written out by hand.
# ---------------------------------------------------------------------------


def operator_scene(class_name: str, op_name: str, Lx: int, Ly: int, p: int, **extra):
    if op_name not in OPS:
        raise SystemExit(f"unknown OP {op_name!r}; choose one of {sorted(OPS)}")
    f = OPS[op_name]
    namespace = dict(
        op=staticmethod(stencil_op(f, (Lx, Ly))),
        prime=p,
        initial_array=mult_matrix(f, (Lx, Ly), p).astype(int),
        kernel_grid_shape=(Lx, Ly),
        title_text=f"{op_name} operator — {Lx}x{Ly} torus over GF({p})",
        # Row-by-row elimination is unreadable once the matrix is big.
        batch_row_ops=(Lx * Ly >= 9),
        __doc__=(
            f"{op_name}: f = {f} on an {Lx}x{Ly} torus over GF({p}). "
            f"N_c and the kernel basis are computed on screen."
        ),
    )
    namespace.update(extra)
    return type(class_name, (VisualGFpRREF,), namespace)


# The configurable one: OP / LX / LY / P at the top of this file, or in the env.
OperatorKernel = operator_scene("OperatorKernel", OP_NAME, LX, LY, P)

# Fixed presets, handy for comparing without touching the config block.
SixNeighborL4 = operator_scene("SixNeighborL4", "six", 4, 4, 2)
NewmanMooreL3 = operator_scene("NewmanMooreL3", "nm", 3, 3, 2)
NewmanMooreL3GF3 = operator_scene("NewmanMooreL3GF3", "nm", 3, 3, 3)
TwoRollL3x2 = operator_scene("TwoRollL3x2", "two_roll", 3, 2, 3)


class KernelGF2Ring(VisualGFpRREF):
    """The (1+x) factor on a periodic L=4 ring — worked by hand in
    notebooks/01_gf2_linear_algebra.ipynb.

    Row i of C t = 0 says t_i = t_{i-1}, so the kernel is spanned by the
    all-ones vector: rank 3, N_c = 1, GSD = 2.
    """

    prime = 2
    initial_array = np.array(
        [
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=int,
    )


class KernelGF2TwoFree(VisualGFpRREF):
    """Notebook 01, section 2: 3x4 with R2 = R0 + R1 over GF(2).

    Two free columns, so two independent operator identities: N_c = 2 and
    GSD = 4. This is the case that shows the free-column <-> basis-vector
    pairing with something to disambiguate.
    """

    prime = 2
    initial_array = np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=int,
    )
