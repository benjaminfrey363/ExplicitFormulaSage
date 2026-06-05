# Project Plan

## Project title

Explicit prime-counting functions and zeta-zero experiments

## Mathematical background

The Chebyshev function is defined by

\[
\psi(x)=\sum_{p^k\le x}\log p.
\]

It is connected to the Riemann zeta function through

\[
-\frac{\zeta'}{\zeta}(s)=\sum_{n=1}^{\infty}\frac{\Lambda(n)}{n^s}.
\]

The explicit formula relates \(\psi(x)\) to the zeros of \(\zeta(s)\):

\[
\psi(x)
=
x-\sum_\rho \frac{x^\rho}{\rho}
-\log(2\pi)
-\frac{1}{2}\log(1-x^{-2}).
\]

The goal of this project is to build computational tools for experimenting with these quantities in SageMath.

## Week goals

### Minimum viable goal

- Compute \(\psi(x)\) and \(\vartheta(x)\) exactly for moderate \(x\).
- Add examples and tests.

### Main goal

- Implement a truncated explicit formula for \(\psi(x)\).
- Compare exact values against approximations using finitely many zeta zeros.

### Stretch goals

- Visualize error terms.
- Investigate Sage's built-in zeta-zero functionality.
- Add basic zero-free-region and zero-density estimate visualizations.

## Task ideas

### Task 1: Chebyshev functions

Implement:

- `von_mangoldt(n)`
- `chebyshev_psi(x)`
- `chebyshev_theta(x)`

### Task 2: Zeta zeros

Investigate existing Sage functionality for accessing zeta zeros.

Possible goals:

- wrapper for first `n` zeros,
- wrapper for zeros up to height `T`,
- documentation of installation requirements.

### Task 3: Explicit formula

Implement a function approximating

\[
x-\sum_{|\gamma|\le T}\frac{x^\rho}{\rho}
-\log(2\pi)
-\frac{1}{2}\log(1-x^{-2}).
\]

### Task 4: Plotting

Add plots comparing:

- \(\psi(x)\) and \(x\),
- \(\psi(x)-x\),
- exact \(\psi(x)\) versus explicit-formula approximations.
