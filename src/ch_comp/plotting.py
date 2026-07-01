


from ch_comp import odlyzko_wrapper as ow, chebyshev, classical, chef
import numpy as np
from sage.all import  numerical_approx


# Approximate L2 distance between 2 functions on pre-specified range xs
# Take as arg values of fns on this range, y1s and y2s
def _l2dist_approx(y1s, y2s):
    if len(y1s) != len(y2s):
        raise ValueError("Function output arrays must have the same size")
    return sum([ abs(y1 - y2)**2 for (y1,y2) in zip(y1s,y2s)])



"""
_, xs, cefs = classical.compute_cef(5000,2,100,1000)
_, _, chefs = chef.compute_chef(5000,2,100,1000)
psis = [chebyshev.chebyshev_psi(x) for x in xs]

print(numerical_approx(_l2dist_approx(cefs,psis),30))
print(numerical_approx(_l2dist_approx(chefs,psis),30))
"""



def full_test_suite():
    """
    Full test and plotting suit to compare CEF and CHEF over different
    numbers of sampled zeros. Saves plots and tables

    Returns
    -------
    cef_low_range : .png file
        Plot of cef(n) for n in [100,500,1000,5000], x in [2,1000]
    cef_mid_range : .png file
        Plot of cef(n) for n in [100,500,1000,5000], x in [1000,2000]
    
    """


