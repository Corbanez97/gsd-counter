# Closed form for $N_c(L_x, L_y)$ — the `haah_z` stencil on a torus over GF(2)

This note reports a closed form for the quantity computed by
[scripts/nc_sweep.py](../scripts/nc_sweep.py):

$$N_c(L_x,L_y) \;=\; \dim_{\mathbb F_2}\ker A, \qquad \mathrm{GSD} = 2^{N_c},$$

where $A$ is the $L_xL_y \times L_xL_y$ matrix of the `haah_z` stencil rolled over an
$L_y\times L_x$ periodic lattice. Everything below is for the default operator
(`--op haah_z`, equivalently `haah_x`) at $p=2$.

A reference implementation is [scripts/nc_closed_form.py](../scripts/nc_closed_form.py);
it reproduces the brute force exactly on every $2 \le L_x,L_y \le 40$ (`--verify 40`).

---

## 1. The result

Write the 2-adic factorisation of each side length,

$$L_x = 2^{a} m_x, \qquad L_y = 2^{b} m_y, \qquad m_x, m_y \ \text{odd},$$

and let $\omega \in \mathbb F_4$ satisfy $\omega^2+\omega+1=0$. Define

$$\boxed{\;N(m_x,m_y) \;=\; \deg \gcd_{\mathbb F_4[y]}\Bigl(\, y^{m_y}+1 \;,\; (\omega y + \omega^2)^{m_x} + 1 \,\Bigr).\;}$$

Then

$$\boxed{\;N_c(L_x,L_y) \;=\; 2\,\min\!\left(2^{a},\,2^{b}\right)\, N(m_x,m_y) \;-\; \delta_{a,b}\;}$$

where $\delta_{a,b}=1$ if $a=b$ (i.e. $v_2(L_x)=v_2(L_y)$) and $0$ otherwise.

Three things are worth emphasising:

* **The dependence on $L_x$ and $L_y$ splits completely.** The 2-part enters only through
  $\min(2^a,2^b)$ and the flag $\delta_{a,b}$; the odd part enters only through $N(m_x,m_y)$.
* **$N$ is an honest closed form**, not a rank computation: it is the degree of a gcd of two
  explicit polynomials over $\mathbb F_4$, computable in $O(m_y \log m_x)$ field operations
  instead of the $O((L_xL_y)^3)$ Gaussian elimination in `nc_sweep.py`. $N \ge 1$ always,
  so $N_c \ge 1$ and the ground space is never trivial.
* **$N$ is genuinely number-theoretic.** It is a cyclotomic point count (§4), and there is
  no elementary formula for it in terms of $m_x,m_y$ alone — this is the source of the
  notorious UV/IR-sensitivity of fracton GSD. What *is* elementary is everything else.

---

## 2. Where it comes from

### 2.1 The operator is a polynomial

`HAAH_Z_SHIFTS` lists offsets $(d_2,d_1)=(\text{row},\text{col})$; identifying a shift by
$d_1$ columns with $x$ and by $d_2$ rows with $y$, the stencil is the polynomial

$$f(x,y) \;=\; 1 + x + x^2 + y + xy + y^2 \;\in\; \mathbb F_2[x,y],$$

and $A$ is multiplication by $f$ in the group ring

$$R_L \;=\; \mathbb F_2[x,y]\big/\bigl(x^{L_x}-1,\; y^{L_y}-1\bigr),$$

so $N_c = \dim_{\mathbb F_2}\ker(f\cdot)$ on $R_L$. (This is the "rolling" picture of
[gsd_constraints_and_symmetries.md](gsd_constraints_and_symmetries.md) §7.1.)

### 2.2 The ring splits; only points of the curve $f=0$ contribute

Over $\mathbb F_2$, $x^{L_x}-1=(x^{m_x}-1)^{2^{a}}$ with $x^{m_x}-1$ squarefree. Hensel
lifting splits $R_L$ into local factors indexed by pairs $(\zeta,\eta)$ with
$\zeta^{m_x}=\eta^{m_y}=1$, each of the form $F[s,t]/(s^{A},t^{B})$ with

$$A = 2^{a}, \qquad B = 2^{b}, \qquad s = x-\zeta,\quad t = y-\eta,$$

$F$ the residue field. Multiplication by $f$ is invertible on a factor unless
$f(\zeta,\eta)=0$, so

$$N_c \;=\; \sum_{\substack{\zeta^{m_x}=\eta^{m_y}=1\\ f(\zeta,\eta)=0}} \kappa(\zeta,\eta),$$

with $\kappa$ a *local multiplicity* depending only on $(A,B)$ and on the local shape of $f$.
Since $f$ is quadratic, its local expansion is exact:

$$f(\zeta+s,\ \eta+t) \;=\; f(\zeta,\eta) \;+\; (1+\eta)\,s \;+\; (1+\zeta)\,t \;+\; \bigl(s^2+st+t^2\bigr).$$

### 2.3 The curve has exactly one singular point, $(1,1)$

$f(1,y) = 1+y^2 = (1+y)^2$, so the only solution with $\zeta=1$ is $(1,1)$, and symmetrically
for $\eta=1$. Hence:

* **$(\zeta,\eta) = (1,1)$.** Both partials $1+\eta$, $1+\zeta$ vanish and
  $f = s^2+st+t^2$ — a singular (nodal) point, present for *every* $L_x,L_y$.
* **Every other point is smooth**, with both linear coefficients nonzero.

### 2.4 Local multiplicities

Direct computation of $\dim\ker$ on $F[s,t]/(s^A,t^B)$ for $A,B$ powers of two gives

| point | local $f$ | $\kappa$ |
|---|---|---|
| smooth | $\alpha s + \beta t + (s^2+st+t^2)$, $\alpha\beta \ne 0$ | $\min(A,B)$ |
| singular $(1,1)$ | $s^2+st+t^2$ | $2\min(A,B) - \delta_{A,B}$ |

The smooth case has a one-line proof: $\partial f/\partial t = \beta$ is a unit, so in
$F[[s,t]]$ Weierstrass gives $f = (\text{unit})\cdot(t - \varphi(s))$ with $\varphi(0)=0$;
if $B \ge A$ then $\varphi(s)^{B}\equiv 0 \bmod s^{A}$ (char 2, $B=2^b$), so
$t \mapsto t+\varphi(s)$ is an automorphism of $F[s,t]/(s^A,t^B)$ and
$\dim\ker(f\cdot)=\dim\operatorname{Ann}(t) = A = \min(A,B)$; swap roles if $A > B$.
The singular case follows from the $\mathbb F_4$-factorisation
$s^2+st+t^2=(s+\omega t)(s+\omega^2 t)$ of two such "smooth" factors, and was verified
directly for all $A,B \in \{1,2,4,8,16\}$:

```
D(A,B) =  A\B    1    2    4    8   16
            1    1    2    2    2    2
            2    2    3    4    4    4
            4    2    4    7    8    8
            8    2    4    8   15   16
           16    2    4    8   16   31
```

Putting $P = \#\{(\zeta,\eta)\}$ for the total number of points,

$$N_c = (P-1)\min(A,B) \;+\; \bigl(2\min(A,B)-\delta_{A,B}\bigr) \;=\; \min(A,B)\,(P+1) - \delta_{A,B}.$$

### 2.5 The conic factors over $\mathbb F_4$

The homogenisation of $f$ is $Q = x^2+y^2+z^2+xy+yz+zx$, whose polar form has radical
$(1:1:1)$ — and $Q(1,1,1)=0$, so the conic is *degenerate*: it is a pair of conjugate lines
meeting at $(1:1:1)$. Explicitly,

$$\boxed{\;f(x,y) \;=\; \bigl(x+\omega y+\omega^2\bigr)\bigl(x+\omega^2 y+\omega\bigr) \quad\text{over } \mathbb F_4.\;}$$

So the "fractal" Haah stencil is, algebraically, **two straight lines over $\mathbb F_4$**.
Frobenius ($u\mapsto u^2$) swaps the two branches and preserves $\mu_{m_x}\times\mu_{m_y}$,
and the branches meet only at $(1,1)$, hence

$$P(m_x,m_y) = 2N(m_x,m_y) - 1, \qquad
N = \#\bigl\{\eta \in \mu_{m_y} \;:\; \omega\eta+\omega^2 \in \mu_{m_x}\bigr\}.$$

Because $y^{m_y}-1$ is squarefree, that count is exactly the gcd degree in §1. Substituting
$P+1 = 2N$ into §2.4 gives the boxed formula.

---

## 3. Explicit special cases

Let $\mu = \min(2^{a},2^{b})$.

| regime | $N_c$ | GSD |
|---|---|---|
| **Generic** ($N=1$; see §4) | $2\mu - \delta_{a,b}$ | $2^{2\mu-\delta_{a,b}}$ |
| $L_x, L_y$ both odd, generic | $1$ | $2$ |
| $L_x = L_y = 2^k$ | $2^{k+1}-1 = 2L-1$ | $2^{2L-1}$ |
| $L_x = 2^{a}$, $L_y = 2^{b}$, $a \ne b$ | $2^{\min(a,b)+1}$ | $2^{2\min(L_x,L_y)}$ |
| $L_x = L_y = L = 2^{a}m$ | $2^{a}\bigl(P(m,m)+1\bigr)-1 = 2^{a+1}N(m,m)-1$ | — |
| $L_x = L_y = 2^{a}(2^{k}-1)$, $k$ **even** | $2^{a}\bigl(2^{k+1}-4\bigr)-1$ | — |
| $L_x = L_y = 2^{a}(2^{k}-1)$, $k$ **odd** | $2^{a+1}-1$ | — |

The $k$ even row is the maximal-degeneracy family: $\mu_{2^k-1} = \mathbb F_{2^k}^\times$ and
$3 \mid 2^k-1$, so the branch condition becomes $u+v=1$ with $u,v \in \mathbb F_{2^k}^\times$,
giving $N = 2^{k}-2$. It reproduces $N_c(15,15)=27$, $N_c(63,63)=123$, $N_c(255,255)=507$,
$N_c(1023,1023)=2043$, $N_c(4095,4095)=8187$ — i.e. $N_c \approx 2L$, the same linear growth
as the pure powers of two, and the largest possible.

Two consequences worth stating separately:

* **$N_c$ grows at most linearly in $L$.** $N_c \le 2\min(L_x,L_y)$ always, so
  $\mathrm{GSD} \le 2^{2\min(L_x,L_y)}$: sub-extensive, as expected for a fracton-like code.
* **A single change of parity destroys the degeneracy.** $N_c(2^k,2^k)=2^{k+1}-1$ but
  $N_c(2^k,\,2^k\pm 1)$ collapses to $2$ (e.g. $N_c(256,256)=511$ while $N_c(256,255)=2$).

---

## 4. When is $N > 1$? (the number-theoretic part)

$N=1$ means the only point of the curve on the torus $\mu_{m_x}\times\mu_{m_y}$ is the
singular one, $(1,1)$. This is the *generic* situation.

**Sufficient criterion for $N=1$.** If $d=\operatorname{lcm}\bigl(\mathrm{ord}_{m_x}(2),\,
\mathrm{ord}_{m_y}(2)\bigr)$ is **odd**, then $\mathbb F_4 \not\subset \mathbb F_{2^d}$; writing
$\xi + \omega\eta + \omega^2 = 0$ in the basis $\{1,\omega\}$ of $\mathbb F_{2^{2d}}$ over
$\mathbb F_{2^d}$ forces $\xi=\eta=1$. Hence $N=1$ and $N_c = 2\mu-\delta_{a,b}$.
(E.g. any $m$ with $m \mid 2^k-1$, $k$ odd: $m=7,31,127,73,\dots$)

**Sufficient criterion for $N > 1$.** If $3 \mid \gcd(m_x,m_y)$ then $\omega \in \mu_{m_x}\cap\mu_{m_y}$
and $(\xi,\eta)=(\omega,\omega^2)$ lies on the branch, so $N \ge 2$, $P \ge 3$. This is why every
$L$ divisible by 3 shows enhanced degeneracy in `data/haah_z.csv`.

**Neither criterion is necessary.** The set $\{m : N(m,m)>1\}$ is the set of multiples of the
*levels* $\ell = \operatorname{lcm}\bigl(\mathrm{ord}(\eta),\mathrm{ord}(\omega\eta+\omega^2)\bigr)$
of the non-singular points. The smallest levels are

$$\ell \;=\; 3,\; 85 = 5\cdot 17,\; 91 = 7\cdot 13,\; 341 = 11\cdot 31,\; 1387 = 19\cdot 73,\;\dots$$

so for $m$ coprime to 3, the first $m$ with $N(m,m)>1$ are $85$ ($N_c = 49$) and $91$
($N_c = 25$) — sizes at which the brute-force sweep is already expensive but the gcd formula
is instant. A scan of all odd $m<3000$ coprime to 3 with $N(m,m)>1$ gives exactly the
multiples of $85, 91, 341, 1387$ in that range.

**Size of $N$.** With $\mu_{m_x},\mu_{m_y} \subset \mathbb F_q$, $q=2^d$, expanding the coset
indicators in multiplicative characters writes $N$ as an average of Jacobi sums,

$$N \;=\; \frac{1}{e_xe_y}\sum_{\chi,\psi} J(\chi,\psi), \qquad e_x=\tfrac{q-1}{m_x},\ e_y=\tfrac{q-1}{m_y},$$

so by Weil $N = \dfrac{m_xm_y}{q} + O(\sqrt q)$. This is the closed form in the analytic sense;
it also explains the erratic look of the data: $N_c$ tracks $m_xm_y/2^{\,\mathrm{ord}}$, which
jumps whenever the multiplicative order of 2 changes.

---

## 5. Tables

$N_c(L_x,L_y)$, $2 \le L_x,L_y \le 16$ (brute force, `data/haah_z_rect.csv`):

| Ly\Lx | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **2** | 3 | 2 | 4 | 2 | 3 | 2 | 4 | 2 | 3 | 2 | 4 | 2 | 3 | 2 | 4 |
| **3** | 2 | 3 | 2 | 1 | 4 | 1 | 2 | 3 | 2 | 1 | 4 | 1 | 2 | 3 | 2 |
| **4** | 4 | 2 | 7 | 2 | 4 | 2 | 8 | 2 | 4 | 2 | 7 | 2 | 4 | 2 | 8 |
| **5** | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 9 | 2 |
| **6** | 3 | 4 | 4 | 2 | 7 | 2 | 4 | 4 | 3 | 2 | 8 | 2 | 3 | 4 | 4 |
| **7** | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 |
| **8** | 4 | 2 | 8 | 2 | 4 | 2 | 15 | 2 | 4 | 2 | 8 | 2 | 4 | 2 | 16 |
| **9** | 2 | 3 | 2 | 1 | 4 | 1 | 2 | 3 | 2 | 1 | 4 | 1 | 2 | 3 | 2 |
| **10** | 3 | 2 | 4 | 2 | 3 | 2 | 4 | 2 | 3 | 2 | 4 | 2 | 3 | 10 | 4 |
| **11** | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 |
| **12** | 4 | 4 | 7 | 2 | 8 | 2 | 8 | 4 | 4 | 2 | 15 | 2 | 4 | 4 | 8 |
| **13** | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 |
| **14** | 3 | 2 | 4 | 2 | 3 | 2 | 4 | 2 | 3 | 2 | 4 | 2 | 3 | 2 | 4 |
| **15** | 2 | 3 | 2 | 9 | 4 | 1 | 2 | 3 | 10 | 1 | 4 | 1 | 2 | 27 | 2 |
| **16** | 4 | 2 | 8 | 2 | 4 | 2 | 16 | 2 | 4 | 2 | 8 | 2 | 4 | 2 | 31 |

The odd-part input $P(m,m)=2N(m,m)-1$:

| $m$ | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 | 21 | 23 | 25 | 27 | 33 | 45 | 63 | 85 | 91 | 93 | 105 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $P$ | 1 | 3 | 1 | 1 | 3 | 1 | 1 | 27 | 1 | 1 | 15 | 1 | 1 | 3 | 3 | 27 | 123 | 49 | 25 | 63 | 39 |

(all odd $m \le 31$ not listed have $P=1$.)

---

## 6. Numerical validation

| check | result |
|---|---|
| Full grid $2 \le L_x,L_y \le 48$ (`--rect`), 2209 lattices, brute force vs. formula | 2209/2209 agree |
| `scripts/nc_closed_form.py --verify 40` (self-contained rerun) | 1521/1521 agree |
| $\gcd$-degree formula for $N$ vs. brute force on all odd $1\le m_x,m_y\le47$ | 576/576 agree |
| Same, cross-checked against an independent $\mathbb F_{2^D}$ root-enumeration | 576/576 agree |
| Asymmetric spot checks: $(64,48),(63,64),(85,32),(91,16),(120,45),(93,31),(51,68),(56,105),(60,96)$ | all agree |
| Square sweep to $L=128$ (`data/haah_z.csv`, pre-existing) | all agree |
| Same formula applied to `--op haah_x` (transpose), $2\le L\le 24$ | 529/529 agree |

Raw data: [data/haah_z.csv](../data/haah_z.csv) (squares to 128),
[data/haah_z_rect.csv](../data/haah_z_rect.csv) (full $2..48$ grid).

---

## 7. Scope

* The derivation is specific to **`haah_z`/`haah_x` at $p=2$**. The `-p` flag of
  `nc_sweep.py` changes the base field and $f$ factors differently there
  (the split $\mathbb F_4$-conic argument is a characteristic-2 statement).
* The *method*, however, is fully general for any single-species translation-invariant
  stencil: (i) write the stencil as $f$; (ii) find the points of $f=0$ on
  $\mu_{m_x}\times\mu_{m_y}$; (iii) attach the local multiplicity of each point in
  $F[s,t]/(s^{p^a},t^{p^b})$. The 2-adic/odd split of §1 is a consequence of
  $x^{p^a m}-1 = (x^{m}-1)^{p^a}$ and holds for any $f$.
* In the convention of [gsd_constraints_and_symmetries.md](gsd_constraints_and_symmetries.md)
  §1, where the code carries both an $X$ and a $Z$ sector, the degeneracy is
  $\mathrm{GSD} = 2^{2N_c}$; `nc_sweep.py` reports the single-sector $\mathrm{GSD}=2^{N_c}$.
