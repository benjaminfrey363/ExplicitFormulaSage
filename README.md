# Sage Days Prime Counting Experiments

This repository contains SageMath experiments related to explicit prime-counting estimates, Chebyshev functions, zeta zeros, and explicit formulae.

Additionally, it includes an attempt to implement Chirre and Helfgott's paper "Optimal bounds for sums of non-negative arithmetic functions" in sage.

The project is motivated by explicit estimates for the Chebyshev function

$\psi(x)=\sum_{p^k\le x}\log p,$

and by the use of explicit formulae, partial verification of the Riemann Hypothesis, zero-density estimates, and zero-free regions.


## Repository structure

```text
src/primecounting/     Core Python/Sage code for prime counting
src/perron/            Core Python/Sage code for Perron-type implementations
src/ch-framework/      Core Python/Sage code for CH paper implementation
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


## Example usage of src/primecounting:

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


## Example usage of src/ch_comp

ch_comp contains practical and "optimized" implementations of Chirre and Helfgott's explicit formula and the classical explicit formula, tests to compare the runtime and approximation error of the two compared to actual values of $\psi(x)$, and plotting functions.


## Example usage of src/ch-framework

ch-framework contains implementations of the framework described in Chirre and Helfgott's paper "Optimal bounds for the sums of nonnegative arithmetic functions".
This includes an implementation of Chirre and Helfgott's Perron-type formula, smooth sum identity, integral bounds, contour shift, and full explicit formula.
Each of these parts can be seen as a "step" in the CH paper, and corresponds to an
identity which is proven true.
These implementations include tests (contained in examples folder) which can be used
to measure the error in each identity when certain variables are truncated/changed.
To run such a test, add desired function to body of test .sage file and run:

e.g. in test_weighted_perron.sage:

```bash
(body)

test_weighted_perron_with_zeta_box()
```

and run:

```bash
sage src/ch-framework/examples/test_weighted_perron.sage
```

(TO-DO: outline of using full pipeline)

