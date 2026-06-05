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

## Setting up odlyzko-zeta:

To install database of zeta zeros, on mac/linux use:

```bash
sage -pip install passagemath-database-odlyzko-zeta
```

I also had to symlink the installed binary:

```bash
mkdir -p "$HOME/Library/Application Support/odlyzko"
ln -s \
  "$HOME/Library/SageMath-10-9/lib/python3.14/site-packages/sage_wheels/share/odlyzko/zeros.sobj" \
  "$HOME/Library/Application Support/odlyzko/zeros.sobj"
```
