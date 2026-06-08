
'''
PerronsFormula.ipynb
Exploratory notebook studying Perron's formula in sage, working towards implementing 
Chirre's Perron-type formula.
'''

def perron_integral(F, x, c, T):
    # determine integrand
    def integrand(t):
        s = c + I*t
        return F(s) * x**s / s

    real_part = numerical_integral(lambda t: real(integrand(t)), -T, T)[0]
    imag_part = numerical_integral(lambda t: imag(integrand(t)), -T, T)[0]

    return (real_part + I*imag_part) / (2*pi)
