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

### 1

- Compute \(\psi(x)\) and \(\vartheta(x)\) exactly for moderate \(x\).
- Add examples and tests.

### 2

- Implement a truncated explicit formula for \(\psi(x)\).
- Compare exact values against approximations using finitely many zeta zeros.

### 3

- Visualize error terms.
- Investigate Sage's built-in zeta-zero functionality.
- Add basic zero-free-region and zero-density estimate visualizations.

