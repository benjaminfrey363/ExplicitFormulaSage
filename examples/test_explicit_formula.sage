# examples/test_explicit_formula.sage

load("src/primecounting/chebyshev.py")
load("src/primecounting/zeta_zeros.py")
load("src/primecounting/explicit_formula.py")

print("First five zeta zero ordinates:")
gammas = first_zeta_zero_imaginary_parts(5)
print(gammas)
print()

x = 100
print(f"Exact psi({x}):")
print(chebyshev_psi(x))
print()

for n in [1, 2, 5, 10, 20, 50, 100]:
    result = compare_with_exact_psi(x, n_zeros=n)
    print(f"Using {n} zeros:")
    print(f"  approx psi({x}) = {result['approx_psi']}")
    print(f"  exact  psi({x}) = {N(result['exact_psi'])}")
    print(f"  error          = {N(result['error'])}")
    print(f"  abs error      = {N(result['absolute_error'])}")
    print()

print("Using zeros up to height T = 50:")
result = compare_with_exact_psi(x, T=50)
print(result)