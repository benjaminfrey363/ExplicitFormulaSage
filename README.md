# Sage Days Prime Counting Experiments

This repository contains SageMath experiments related to explicit prime-counting estimates, Chebyshev functions, zeta zeros, and explicit formulae.

The project is motivated by explicit estimates for the Chebyshev function

$\psi(x)=\sum_{p^k\le x}\log p,$

and by the use of explicit formulae, partial verification of the Riemann Hypothesis, zero-density estimates, and zero-free regions.

## Goals

The main goals for Sage Days are:

1. Implement or identify existing SageMath functionality for computing Chebyshev functions.
2. Build small tools for experimenting with truncated explicit formulae for $\psi(x)$.
3. Use available zeta-zero data in SageMath where possible.
4. Visualize the effect of including more zeros in the explicit formula.
5. Explore how partial RH verification, zero-density estimates, and zero-free regions might be represented computationally.

## Repository structure

```text
src/primecounting/      Core Python/Sage code
notebooks/             Exploratory notebooks
examples/              Small Sage scripts demonstrating functionality
tests/                 Basic tests
docs/                  Project notes and task planning
```