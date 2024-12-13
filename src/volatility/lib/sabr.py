import numpy as np

NUM_EPS = 1e-4
RHO_BOUNDS = (-1+NUM_EPS, 1-NUM_EPS)
ALPHA_BOUNDS = VOLVOL_BOUNDS = (NUM_EPS, None)

# https://en.wikipedia.org/wiki/SABR_volatility_model
# 0 <= beta <= 1
# -1 <= rho <= 1
# volvol > 0
def get_vol(forward_price: float, strike: float, tau: float,
            alpha: float, beta: float, rho: float, volvol: float,
            shift: float = 0) -> float:
    lfk = np.log((forward_price + shift) / (strike + shift))
    fmid = np.sqrt((forward_price + shift) * (strike + shift))
    fmid_b = fmid ** (1 - beta)
    if beta < NUM_EPS:
        zeta = volvol / alpha * (forward_price - strike)
        # zeta = volvol / alpha * fmid_b * lfk
    elif 1-beta < NUM_EPS:
        zeta = volvol / alpha * lfk * (1 + np.log(fmid_b) * (1-beta))
    else:
        zeta = volvol / alpha * ((forward_price + shift)**(1-beta) - (strike + shift)**(1-beta)) / (1-beta)
    if abs(zeta) < NUM_EPS:
        x = alpha / fmid_b
    else:
        if 1-rho < NUM_EPS:
            d_zeta = np.log(1 + zeta / abs(1 - zeta))
        else:
            d_zeta = np.log((np.sqrt(1 - 2*rho*zeta + zeta**2) + zeta - rho) / (1-rho))
        x = alpha * zeta / (fmid_b * d_zeta)
    
    a = ((1-beta) * alpha / fmid_b)**2 / 24
    b = rho * volvol * beta * alpha / (fmid_b * 4)
    c = (2 - 3 * rho**2) * volvol**2 / 24
    g = (1-beta) * lfk

    return x * (1 + (a + b + c) * tau) / (1 + g**2 / 24)  # + g**4 / 1920

# https://github.com/ynouri/pysabr/blob/master/pysabr/models/hagan_2002_lognormal_sabr.py#L86
def get_alpha(vol_atmf: float, forward_price: float, tau: float, volvol: float, beta: float, rho: float) -> float:
    f_b = forward_price ** (1-beta)
    a = (1-beta)**2 * tau / (f_b**3 * 24)
    b = rho * volvol * beta * tau / (f_b**2 * 4)
    c = (1 + (2 - 3 * rho**2) * volvol**2 * tau / 24) / f_b
    coeff = [a, b, c, -vol_atmf]
    roots = np.roots(coeff)
    roots_real = np.extract(np.isreal(roots), np.real(roots))
    first_guess = vol_atmf * f_b
    i_min = np.argmin(np.abs(roots_real - first_guess))
    return roots_real[i_min]
