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

Will check install for Windows.


## Example usage:

Within SageMath environment,

```bash
sage: load('src/primecounting/chebyshev.py')
sage: von_mangoldt(64)
0.6931471805599453
sage: von_mangoldt(27)
1.0986122886681098
sage: von_mangoldt(10)
0
sage: chebyshev_psi(1000)
996.6809122471752
sage: chebyshev_theta(100)
83.72839039906393
sage: load('src/primecounting/zeta_zeros.py')
sage: first_zeta_zero_imaginary_parts(1)
[14.134725142]
sage: first_zeta_zero_imaginary_parts(10)
[14.134725142,
 21.022039639,
 25.01085758,
 30.424876126,
 32.935061588,
 37.586178159,
 40.918719012,
 43.327073281,
 48.005150881,
 49.773832478]
sage: zeta_zeros_up_to_height(40)
[14.134725142,
 21.022039639,
 25.01085758,
 30.424876126,
 32.935061588,
 37.586178159]
```

