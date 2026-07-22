

from sage.all import I, cot, pi



def theta_T_sigma(T, sigma):
    def theta(s):
        return 1 - ((s - sigma)/(I*T))
    return theta


def c_T_sigma(T, sigma):
    theta_eval = theta_T_sigma(T,sigma)(1 + I*T)
    return theta_eval * cot(pi * theta_eval)


def omega_T_sigma(T, sigma):
    c = c_T_sigma(T, sigma)
    theta = theta_T_sigma(T, sigma)
    def omega(s):
        return (- theta(s) * cot(pi*theta(s)) + c)
    return omega


def ch_weight_T_xi(T, xi):
    def ch_weight(s):
        omega_term = omega_T_sigma(T,0)(s)
        theta_term = theta_T_sigma(T,1)(s)
        raw_weight = omega_term + xi*theta_term*I
        return abs(raw_weight)
    return ch_weight
