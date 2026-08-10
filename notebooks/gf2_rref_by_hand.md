# RREF and kernels over $GF(2)$, done by hand

Companion note to `fractal_constraints_Nc.ipynb`. The notebook computes $N_c = \dim\ker A$ with `gf2_rref` / `gf2_nullspace` and never shows the arithmetic. This note does the arithmetic on the smallest interesting case — a $2\times 2$ transformation — slowly enough that you can check every step, then connects it back to the lattice operator.

---

## 0. The only rule you need

$GF(2)$ is $\{0,1\}$ with addition and multiplication mod 2:

| $+$ | 0 | 1 |     | $\times$ | 0 | 1 |
|---|---|---|---|---|---|---|
| **0** | 0 | 1 |     | **0** | 0 | 0 |
| **1** | 1 | 0 |     | **1** | 0 | 1 |

Addition is XOR. Three consequences drive everything below:

1. **$1 + 1 = 0$**, so every element is its own additive inverse: $-a = a$. *Therefore "subtract row $r$" and "add row $r$" are the same operation.* Gaussian elimination over $GF(2)$ has no minus signs anywhere.
2. **The only nonzero scalar is $1$.** There is no "divide the pivot row by the pivot" step — a pivot is already $1$ the moment it is nonzero.
3. **A row operation is idempotent**: doing $R_i \mathrel{{+}{=}} R_j$ twice returns you to the start.

So textbook Gaussian elimination survives intact, minus two steps (rescaling, sign tracking). What is left is: find a pivot, swap it up, XOR it into every other row that has a 1 in that column.

---

## 1. The worked example

Take

$$
A \;=\; \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \quad\text{over } GF(2).
$$

This is not an arbitrary choice. It is the matrix of the polynomial factor $\,1+x\,$ acting on a periodic ring of $L=2$ sites — one of the three factors in $xy\,P(x,y) = (1+x)(1+y)(x+y) \bmod 2$ from Section 8 of the notebook. Column $k$ is the image of the delta function at site $k$: site $0$ maps to $\{0,1\}$, site $1$ maps to $\{1,0\}$ (wrapping), so both columns are $\binom{1}{1}$.

We want

$$
\ker A = \{\, t \in GF(2)^2 \;:\; A\,t = 0 \,\}.
$$

Writing $t = (t_0, t_1)^{\mathsf T}$, the linear system is

$$
\begin{aligned}
\text{row 0:}\quad & t_0 + t_1 = 0 \\
\text{row 1:}\quad & t_0 + t_1 = 0
\end{aligned}
$$

— visibly one equation stated twice. RREF is the mechanical procedure that discovers that redundancy for you.

### 1.1 Elimination, column by column

The notebook's `gf2_rref` walks columns left to right, holding a "next row to fill" counter `r` that starts at 0. I will annotate each step with what the code is doing.

**Column 0.** Search rows $r=0,1$ for a 1 in column 0. Row 0 has one, so $\text{piv}=0$.

*Swap:* $\text{piv} = r = 0$, so the swap `M[[r, piv]] = M[[piv, r]]` is a no-op.

*Eliminate:* every **other** row with a 1 in column 0 gets the pivot row added to it. Row 1 qualifies:

$$
R_1 \leftarrow R_1 + R_0 = (1,1) + (1,1) = (1{+}1,\; 1{+}1) = (0,0).
$$

This is the step where $1+1=0$ does all the work — over $\mathbb{R}$ you would have written $R_1 \leftarrow R_1 - R_0$; over $GF(2)$ the minus is unnecessary because $-R_0 = R_0$.

The matrix is now

$$
\begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix},
\qquad \text{pivots} = [\,0\,], \qquad r \to 1 .
$$

Note that elimination runs over *all* rows other than the pivot row, not just rows below it. That is what makes the output *reduced* row echelon form (zeros above pivots too) rather than plain echelon form. Here there is nothing above row 0, so the distinction is invisible; it matters in §5.

**Column 1.** Search rows $r=1$ (only) for a 1 in column 1. Row 1 is $(0,0)$ — no pivot found. The code hits `continue` and column 1 is left alone.

**Result.**

$$
R = \operatorname{rref}(A) = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix},
\qquad
\text{pivot columns } P = \{0\},
\qquad
\text{free columns } F = \{1\}.
$$

$$
\operatorname{rank} A = |P| = 1 .
$$

### 1.2 Reading the kernel off the RREF

Rank–nullity over $GF(2)$ reads exactly as over $\mathbb{R}$:

$$
\dim\ker A = n_{\text{cols}} - \operatorname{rank} A = 2 - 1 = 1 .
$$

The recipe: **one basis vector per free column.** For free column $f$, set $t_f = 1$, set all *other* free variables to $0$, then solve the pivot variables from the RREF rows.

Here $F = \{1\}$, so there is exactly one basis vector. Put $t_1 = 1$. Row 0 of $R$ says

$$
1\cdot t_0 + 1\cdot t_1 = 0 \;\;\Longrightarrow\;\; t_0 = -R_{0,1}\,t_1 = -1 = 1 .
$$

That minus sign is where the code's one surprising line comes from:

```python
for i, p in enumerate(pivots):
    v[p] = R[i, f]          # no minus sign — because -1 == +1 in GF(2)
```

Over $\mathbb{R}$ this line would have to be `v[p] = -R[i, f]`. Over $GF(2)$ the negation is the identity map, so it is dropped. Everything else about the back-substitution is standard.

$$
\boxed{\;\ker A = \operatorname{span}_{GF(2)}\{(1,1)\} = \{(0,0),\,(1,1)\}\;}
$$

The kernel has $2^{\dim} = 2^1 = 2$ elements — over $GF(2)$ a $d$-dimensional subspace is a *finite* set of $2^d$ vectors, which is precisely why $\mathrm{GSD} = 2^{N_c}$ counts something.

### 1.3 Verify

$$
A\begin{pmatrix}1\\1\end{pmatrix}
= \begin{pmatrix} 1\cdot1 + 1\cdot 1 \\ 1\cdot 1 + 1\cdot 1\end{pmatrix}
= \begin{pmatrix} 1+1 \\ 1+1 \end{pmatrix}
= \begin{pmatrix} 0 \\ 0 \end{pmatrix}. \;\checkmark
$$

And $A(0,0)^{\mathsf T} = 0$ trivially. The other two vectors of $GF(2)^2$ are *not* in the kernel: $A(1,0)^{\mathsf T} = (1,1)^{\mathsf T} \ne 0$ and likewise for $(0,1)$. So the kernel is exactly 2 of the 4 configurations — half, as a codimension-1 subspace must be.

**Physical reading.** $t = (1,1)$ means "apply the operator at *both* sites." The claim $A t = 0$ says that product is the identity — a redundant constraint, an operator identity. That is one unit of $N_c$.

---

## 2. Code walkthrough: from a matrix to *every* kernel vector

§1 had a single free column, so the recipe had nothing to disambiguate and the code was easy to skim past. This section runs the notebook's two functions statement by statement on an example with **two** free columns, then closes the last gap: `gf2_nullspace` returns a *basis*, while the physics wants the *whole* kernel. Getting from one to the other is a third short function the notebook never spells out.

The example:

$$
A \;=\; \begin{pmatrix} 1&1&0&1 \\ 0&1&1&0 \\ 1&0&1&1 \end{pmatrix},
\qquad\text{note that } R_2 = R_0 + R_1 \text{ already, over } GF(2).
$$

Three equations, only two of them independent — so expect $\operatorname{rank} A = 2$ and $\dim\ker A = 4 - 2 = 2$. The point of choosing a wide matrix ($3$ rows, $4$ columns) is that wide matrices are exactly where kernels come from, and the notebook's $L^2\times L^2$ operator is square but rank-deficient for the same reason: repeated rows.

### 2.1 `gf2_rref`, statement by statement

```python
def gf2_rref(M):
    """Row-reduced echelon form over GF(2). Returns (R, pivot_columns)."""
    M = M.clone() % 2          # (a) never mutate the caller's matrix; force entries into {0,1}
    rows, cols = M.shape
    pivots, r = [], 0          # (b) `pivots` records WHICH columns got one; `r` = next row to fill
    for col in range(cols):    # (c) sweep columns left to right
        piv = None
        for rr in range(r, rows):          # (d) look only in rows not yet used as pivot rows
            if M[rr, col] == 1:
                piv = rr
                break                      # (e) first hit wins: any 1 is as good a pivot as any other
        if piv is None:
            continue                       # (f) free column: leave it alone, do NOT advance r
        M[[r, piv]] = M[[piv, r]]          # (g) bring the pivot row up to row r
        for rr in range(rows):             # (h) ALL rows, not just rr > r — this is what makes it *reduced*
            if rr != r and M[rr, col] == 1:
                M[rr] = (M[rr] + M[r]) % 2 # (i) XOR: addition and subtraction in one operation
        pivots.append(col)
        r += 1                             # (j) row r is now spoken for
        if r == rows:
            break                          # (k) out of rows; every remaining column is free
    return M, pivots
```

Four invariants are worth holding onto while reading:

- **`r` counts the pivots found so far**, and it advances *only* at line (j) — never in the `continue` branch. So at every moment `len(pivots) == r`, and rows $0..r-1$ each carry exactly one pivot. That coupling is what makes the pairing in §2.2 correct.
- **`pivots[i]` is the pivot column of row `i`.** The list comes out strictly increasing because the outer loop moves left to right and never revisits a column.
- **Line (h) is the entire difference between echelon and *reduced* echelon form.** Restricting it to `rr > r` would still give the right rank, but would break the one-line back-substitution in §2.2.
- **Line (e) may grab any nonzero row.** Over $\mathbb{R}$ you would hunt for the largest entry to keep the arithmetic stable; over $GF(2)$ the only nonzero entry *is* $1$, there is no rounding error to accumulate, and the choice is genuinely free.

**Trace on $A$.**

`col = 0`, `r = 0`. Rows $0..2$ are scanned and $M_{0,0} = 1$, so `piv = 0` and the swap is a no-op. The rows with a 1 in column 0 other than row 0: just row 2. So $R_2 \leftarrow R_2 + R_0 = (1,0,1,1) + (1,1,0,1) = (0,1,1,0)$.

$$
\begin{pmatrix} 1&1&0&1 \\ 0&1&1&0 \\ 0&1&1&0 \end{pmatrix},
\qquad \texttt{pivots} = [0], \qquad r \to 1 .
$$

`col = 1`, `r = 1`. Scan rows $1..2$; $M_{1,1} = 1$, so `piv = 1` and the swap is again a no-op. Now the rows with a 1 in column 1 other than row 1 are rows 0 **and** 2 — this is the moment line (h) earns its keep, because row 0 sits *above* the pivot row and a below-only loop would skip it. So $R_0 \leftarrow R_0 + R_1 = (1,1,0,1) + (0,1,1,0) = (1,0,1,1)$ and $R_2 \leftarrow R_2 + R_1 = (0,1,1,0) + (0,1,1,0) = (0,0,0,0)$.

$$
R = \begin{pmatrix} 1&0&1&1 \\ 0&1&1&0 \\ 0&0&0&0 \end{pmatrix},
\qquad \texttt{pivots} = [0,1], \qquad r \to 2 .
$$

`col = 2`, `r = 2`. Only row 2 is in range, and $M_{2,2} = 0$: no pivot, so `continue` fires and `r` stays at 2. `col = 3` is the same story. The loop ends with

$$
P = \{0,1\},\qquad F = \{2,3\},\qquad \operatorname{rank} A = 2,\qquad \dim\ker A = 4-2 = 2 .
$$

The dependent row $R_2 = R_0 + R_1$ we spotted by eye has become an explicit row of zeros. That is the general pattern: RREF converts "some rows were redundant" into "some rows are visibly zero", and the count of surviving nonzero rows is the rank.

### 2.2 `gf2_nullspace`, statement by statement

```python
def gf2_nullspace(A):
    """Basis for {t : A t = 0} over GF(2), one vector per row."""
    R, pivots = gf2_rref(A)
    cols = A.shape[1]
    free = [c for c in range(cols) if c not in pivots]   # (a) free columns = complement of the pivots
    basis = []
    for f in free:                                       # (b) exactly ONE basis vector per free column
        v = torch.zeros(cols, dtype=torch.int64)
        v[f] = 1                                         # (c) switch this free variable on, all others off
        for i, p in enumerate(pivots):                   # (d) i = ROW of R, p = that row's PIVOT COLUMN
            v[p] = R[i, f]                               # (e) back-substitution; no minus, since -1 = 1
        basis.append(v)
    return torch.stack(basis) if basis else torch.zeros((0, cols), dtype=torch.int64)
```

**Why line (c) is the whole trick.** Every free column is a variable you are free to choose; every pivot column is a variable the equations then force on you. Setting one free variable to 1 and all the others to 0 isolates a single degree of freedom, which is why the resulting vectors are automatically independent: basis vector $f$ is the only one in the whole list with a 1 in coordinate $f$, so no combination of the others can reproduce it. There is no orthogonalisation step here, and over $GF(2)$ there is no such thing anyway — a nonzero vector can be "orthogonal" to itself, e.g. $(1,1)\cdot(1,1) = 0$.

**Why line (e) is correct.** In *reduced* echelon form, row $i$ has a 1 in its own pivot column $p_i$ and a 0 in every other pivot column; whatever else it carries lives in the free columns. So row $i$ of the system $Rt = 0$ reads

$$
t_{p_i} \;+\; \sum_{f \in F} R_{i,f}\, t_f \;=\; 0
\qquad\Longrightarrow\qquad
t_{p_i} \;=\; \sum_{f \in F} R_{i,f}\, t_f ,
$$

with no minus sign on the right because $-1 = 1$. With only $t_f = 1$ switched on, that sum collapses to the single term $R_{i,f}$ — which is line (e) verbatim. The `enumerate` is doing the row↔pivot-column pairing, and §5 is the case where confusing the two would give you a wrong answer silently.

**Trace.** `pivots = [0, 1]` and `free = [2, 3]`, so two basis vectors.

$f = 2$: start from $v = (0,0,1,0)$; then $i=0$ sets $v_0 = R_{0,2} = 1$ and $i=1$ sets $v_1 = R_{1,2} = 1$. Result $v^{(2)} = (1,1,1,0)$.

$f = 3$: start from $v = (0,0,0,1)$; then $i=0$ sets $v_0 = R_{0,3} = 1$ and $i=1$ sets $v_1 = R_{1,3} = 0$. Result $v^{(3)} = (1,0,0,1)$.

Check both against the **original** $A$, not against $R$ — that is the check the notebook's `validate` performs, and it is the one that catches an elimination bug, because a mistake made during RREF is invisible to $R$ itself:

$$
A\,v^{(2)} = \begin{pmatrix}1{+}1{+}0{+}0\\ 0{+}1{+}1{+}0\\ 1{+}0{+}1{+}0\end{pmatrix} = 0 \;\checkmark
\qquad
A\,v^{(3)} = \begin{pmatrix}1{+}0{+}0{+}1\\ 0{+}0{+}0{+}0\\ 1{+}0{+}0{+}1\end{pmatrix} = 0 \;\checkmark
$$

### 2.3 From the basis to all $2^d$ kernel vectors

`gf2_nullspace` hands back $d = \dim\ker A$ vectors, but $\ker A$ contains $2^d$ of them. Over $GF(2)$ you can close that gap by brute listing, because the only scalars are 0 and 1: a linear combination *is* a choice of subset, and the combination *is* the XOR of that subset.

$$
\ker A \;=\; \Big\{\; \textstyle\sum_{j \in S} b_j \;:\; S \subseteq \{0,\dots,d-1\} \;\Big\},
\qquad |\ker A| = 2^d .
$$

The bit pattern of an integer `mask` in $[0, 2^d)$ names the subset, which turns the sum above into a two-line loop:

```python
def gf2_all_kernel_vectors(A):
    """Every t with A t = 0, as a (2^d, n_cols) tensor. Row `mask` is the XOR of the basis
    vectors whose index bit is set in `mask`; row 0 is the zero vector."""
    B = gf2_nullspace(A)                                # d basis vectors, d = dim ker A
    d, cols = B.shape
    out = torch.zeros((2 ** d, cols), dtype=torch.int64)
    for mask in range(2 ** d):
        for j in range(d):
            if (mask >> j) & 1:                         # is basis vector j in this subset?
                out[mask] = (out[mask] + B[j]) % 2      # XOR it in
    return out
```

Two facts turn this from "a list of some kernel vectors" into "the kernel, exactly once each":

- **Nothing is missed.** Every $t \in \ker A$ is *some* combination of the basis, and over $GF(2)$ every combination is one of these $2^d$ subsets. There are no other coefficients to try.
- **Nothing is repeated.** If two subsets gave the same vector, $\sum_{j\in S} b_j = \sum_{j\in S'} b_j$, then adding both sides gives $\sum_{j \in S \,\triangle\, S'} b_j = 0$, and independence forces the symmetric difference to be empty, i.e. $S = S'$. So `mask` $\mapsto$ kernel vector is a bijection. That bijection *is* the proof of $|\ker A| = 2^d$, and it is why $\mathrm{GSD} = 2^{N_c}$ counts states rather than dimensions.

For the running example $d = 2$, so the full kernel is four vectors:

| `mask` | subset | $t$ | $At$ |
|---|---|---|---|
| `00` | $\varnothing$ | $(0,0,0,0)$ | $0\ \checkmark$ |
| `01` | $\{v^{(2)}\}$ | $(1,1,1,0)$ | $0\ \checkmark$ |
| `10` | $\{v^{(3)}\}$ | $(1,0,0,1)$ | $0\ \checkmark$ |
| `11` | both | $(0,1,1,1)$ | $0\ \checkmark$ |

The last row is the one to sit with: $(1,1,1,0) + (1,0,0,1) = (0,1,1,1)$, which looks nothing like either generator yet annihilates $A$ just as well. Physically, the product of two operator identities is again an operator identity. That closure under XOR is precisely the statement that the constraints form a *space* rather than a list, which is why counting them with $\dim$ — rather than by counting distinct-looking constraints — is the right move.

One caution on scale. Enumeration is fine for the toy above and for $L=2$, and hopeless past that: the notebook's $L=4$ already has $N_c = 10$, so $1024$ vectors, and $L=8$ would be $2^{22}$. Enumerate to build intuition on small cases; for anything real, carry the $d$ basis vectors and report $2^{N_c}$ symbolically. The notebook's cell that prints each `v.reshape(L, L)` is doing exactly that — inspecting the $N_c$ generators, never the $2^{N_c}$ group elements.

---

## 3. All sixteen $2\times2$ matrices over $GF(2)$

There are only $2^4 = 16$ of them, so the entire classification fits in a table. Every one was checked against the notebook's `gf2_rref` / `gf2_nullspace`.

| $A$ | rank | pivot cols | RREF | $\ker A$ basis | $\dim\ker$ |
|---|---|---|---|---|---|
| $\left(\begin{smallmatrix}0&0\\0&0\end{smallmatrix}\right)$ | 0 | — | $\left(\begin{smallmatrix}0&0\\0&0\end{smallmatrix}\right)$ | $(1,0),\,(0,1)$ | 2 |
| $\left(\begin{smallmatrix}0&0\\0&1\end{smallmatrix}\right)$ | 1 | 1 | $\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$ | $(1,0)$ | 1 |
| $\left(\begin{smallmatrix}0&0\\1&0\end{smallmatrix}\right)$ | 1 | 0 | $\left(\begin{smallmatrix}1&0\\0&0\end{smallmatrix}\right)$ | $(0,1)$ | 1 |
| $\left(\begin{smallmatrix}0&0\\1&1\end{smallmatrix}\right)$ | 1 | 0 | $\left(\begin{smallmatrix}1&1\\0&0\end{smallmatrix}\right)$ | $(1,1)$ | 1 |
| $\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$ | 1 | 1 | $\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$ | $(1,0)$ | 1 |
| $\left(\begin{smallmatrix}0&1\\0&1\end{smallmatrix}\right)$ | 1 | 1 | $\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$ | $(1,0)$ | 1 |
| $\left(\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\right)$ | 2 | 0,1 | $I$ | — | 0 |
| $\left(\begin{smallmatrix}0&1\\1&1\end{smallmatrix}\right)$ | 2 | 0,1 | $I$ | — | 0 |
| $\left(\begin{smallmatrix}1&0\\0&0\end{smallmatrix}\right)$ | 1 | 0 | $\left(\begin{smallmatrix}1&0\\0&0\end{smallmatrix}\right)$ | $(0,1)$ | 1 |
| $\left(\begin{smallmatrix}1&0\\0&1\end{smallmatrix}\right)$ | 2 | 0,1 | $I$ | — | 0 |
| $\left(\begin{smallmatrix}1&0\\1&0\end{smallmatrix}\right)$ | 1 | 0 | $\left(\begin{smallmatrix}1&0\\0&0\end{smallmatrix}\right)$ | $(0,1)$ | 1 |
| $\left(\begin{smallmatrix}1&0\\1&1\end{smallmatrix}\right)$ | 2 | 0,1 | $I$ | — | 0 |
| $\left(\begin{smallmatrix}1&1\\0&0\end{smallmatrix}\right)$ | 1 | 0 | $\left(\begin{smallmatrix}1&1\\0&0\end{smallmatrix}\right)$ | $(1,1)$ | 1 |
| $\left(\begin{smallmatrix}1&1\\0&1\end{smallmatrix}\right)$ | 2 | 0,1 | $I$ | — | 0 |
| $\left(\begin{smallmatrix}1&1\\1&0\end{smallmatrix}\right)$ | 2 | 0,1 | $I$ | — | 0 |
| $\left(\begin{smallmatrix}1&1\\1&1\end{smallmatrix}\right)$ | 1 | 0 | $\left(\begin{smallmatrix}1&1\\0&0\end{smallmatrix}\right)$ | $(1,1)$ | 1 |

Two things worth noticing:

- **Only 6 of the 16 are invertible.** Over $GF(2)$, $\det A = ad - bc = ad + bc \bmod 2$, so $A$ is invertible iff $ad + bc = 1$. The invertible ones form $GL_2(\mathbb{F}_2)$, a group of order $(2^2-1)(2^2-2) = 6$ — isomorphic to $S_3$.
- **Determinant is a complete rank test here**, but do not build intuition on it. For the $L^2 \times L^2$ matrices in the notebook you want RREF, which gives rank *and* an explicit kernel basis in one pass; a determinant gives you a single bit.

---

## 4. Back to the lattice operator

### 4.1 The $L=2$ lattice is degenerate

Tempting to reach for $L=2$ as "the $2\times2$ case," but it collapses. On a 2-site ring, a shift by $+1$ and a shift by $-1$ are the *same* permutation. The notebook's six shifts

```
(0,1), (0,-1), (1,0), (-1,0), (-1,1), (1,-1)
```

therefore pair up into three identical pairs, and each pair sums to $0$ mod 2. So for $L=2$:

$$
A = 0_{4\times 4},\qquad \operatorname{rank} A = 0, \qquad N_c = 4 .
$$

Every one of the $2^4 = 16$ configurations is an operator identity. This is consistent with the notebook's $N_c = 3L-2 = 4$, and it is a real data point, but it is useless for learning RREF: there is no elimination to do. That is why §1 uses a single polynomial factor instead of the full six-neighbour operator.

### 4.2 The claim $\dim\ker(1+x) = 1$, for every $L$

Section 8 of the notebook asserts that $\ker(1+x)$ is "always exactly one-dimensional, spanned by the all-ones vector along that direction." The $2\times2$ computation of §1 is the $L=2$ case of that statement. Here is the general argument, which is now just §1 with more columns.

On an $L$-site ring, $1+x$ has circulant matrix $C$ with $C_{i,k} = 1$ iff $i = k$ or $i = k+1$ (mod $L$). Row $i$ of the system $Ct = 0$ says

$$
t_i + t_{i-1} = 0 \;\;\Longleftrightarrow\;\; t_i = t_{i-1}.
$$

The $L$ equations chain all coordinates together, so $t$ is constant; periodicity closes the chain consistently for any $L$ (over $GF(2)$ there is no sign alternation to obstruct it). Hence $\ker(1+x) = \{\mathbf{0}, \mathbf{1}\}$ — dimension exactly 1, always.

RREF sees this as "$L-1$ pivots, one free column." For $L=4$:

$$
C = \begin{pmatrix}
1&0&0&1\\ 1&1&0&0\\ 0&1&1&0\\ 0&0&1&1
\end{pmatrix}
\;\xrightarrow{\ \text{rref}\ }\;
\begin{pmatrix}
1&0&0&1\\ 0&1&0&1\\ 0&0&1&1\\ 0&0&0&0
\end{pmatrix},
\qquad P = \{0,1,2\},\ F = \{3\}.
$$

Back-substituting with $t_3 = 1$: $t_0 = R_{0,3} = 1$, $t_1 = R_{1,3} = 1$, $t_2 = R_{2,3} = 1$, giving $t = (1,1,1,1)$ — the all-ones vector, as promised. The same pattern holds at every $L$: the last column of the RREF is all ones above the zero row.

This is one third of the $3L-2$ story. The remaining work is the overlap counting between $\ker(1+x)$, $\ker(1+y)$ and $\ker(x+y)$ on the 2D lattice, which is the open item in Section 8 of the notebook.

---

## 5. One subtlety the $2\times2$ case cannot show you

With two columns, a free column is always the last one. Then back-substitution is trivially "copy the last column." That is misleading. Consider

$$
M = \begin{pmatrix} 1&1&0 \\ 0&0&1 \end{pmatrix},
$$

which is already in RREF, with $P = \{0, 2\}$ and $F = \{1\}$ — the free column sits **between** two pivot columns.

The recipe is unchanged, but you must pair the $i$-th *row* of $R$ with the $i$-th *pivot column*, not with column $i$:

- $t_1 = 1$ (the free variable).
- Row $i=0$ carries pivot $p=0$: $t_0 = R_{0,1} = 1$.
- Row $i=1$ carries pivot $p=2$: $t_2 = R_{1,1} = 0$.

$$
\ker M = \operatorname{span}\{(1,1,0)\}.
$$

Check: row 0 gives $1 + 1 + 0 = 0\ \checkmark$; row 1 gives $0\ \checkmark$.

That row↔pivot pairing is exactly what `enumerate(pivots)` is doing in `gf2_nullspace`:

```python
for i, p in enumerate(pivots):   # i indexes ROWS of R, p indexes PIVOT COLUMNS
    v[p] = R[i, f]
```

It is also why the elimination loop in `gf2_rref` must clear entries **above** the pivot as well as below (`for rr in range(rows): if rr != r ...`). Plain echelon form would leave nonzeros above the pivots and this one-line back-substitution would be wrong.

---

## 6. Summary of the procedure

Given $A$ over $GF(2)$:

1. **RREF.** Left to right over columns; find any row $\ge r$ with a 1; swap it to row $r$; XOR that row into every other row having a 1 in this column; record the column as a pivot; $r{+}{+}$. No rescaling, no signs.
2. **Rank** $= $ number of pivot columns. **Nullity** $= n_{\text{cols}} - \operatorname{rank}$.
3. **Kernel basis.** One vector per free column $f$: set $t_f = 1$, other free variables $0$, and $t_{p_i} = R_{i,f}$ for each pivot $p_i$ (no minus sign, because $-1 = 1$).
4. **Whole kernel, when you want it.** XOR every subset of the basis: the $2^{\dim\ker A}$ subsets give the $2^{\dim\ker A}$ kernel vectors, each exactly once (§2.3). Only practical for small cases.
5. **Count states.** $|\ker A| = 2^{\dim \ker A}$; in the notebook this is $\mathrm{GSD} = 2^{N_c}$.

Step 3 is where the physics enters: each free column is one independent operator identity, one redundant stabilizer, one factor of 2 in the ground-state degeneracy.
