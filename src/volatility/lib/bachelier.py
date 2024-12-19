import numpy as np
import scipy.stats as scst

from volatility.models.construct_types import OptionMoneynessType

def get_implied_volatility(option_price: float, forward_price: float, strike: float, tau: float,
                    flag: int = 1, discount_factor: float = 1) -> float:
    fwd_premium = option_price / discount_factor
    return (fwd_premium - flag * (forward_price-strike)) / np.sqrt(tau)

def get_premium(forward_price: float, strike: float, tau: float, sigma: float,
                flag: int = 1, discount_factor: float = 1) -> float:
    s_t = sigma * np.sqrt(tau)
    d = (forward_price - strike) / s_t
    return discount_factor * (flag * (forward_price - strike) * 
            scst.norm.cdf(flag * d, loc=0, scale=1) + s_t  * scst.norm.pdf(d, loc=0, scale=1))

def get_delta(forward_price: float, strike: float, tau: float, sigma: float,
              flag: int = 1, discount_factor: float = 1) -> float:
    d = (forward_price - strike) / (sigma * np.sqrt(tau))
    return flag * discount_factor * scst.norm.cdf(flag * d)

def get_vega(forward_price: float, strike: float, tau: float, sigma: float, discount_factor: float = 1) -> float:
    s_t = sigma * np.sqrt(tau)
    d = (forward_price - strike) / s_t
    return discount_factor * scst.norm.pdf(d) * np.sqrt(tau) * 1e-2

def get_theta(forward_price: float, strike: float, tau: float, sigma: float, discount_factor: float = 1) -> float:
    d = (forward_price - strike) / (sigma * np.sqrt(tau))
    return -discount_factor * scst.norm.pdf(d) * sigma / (2 * np.sqrt(tau))

def get_gamma(forward_price: float, strike: float, tau: float, sigma: float, discount_factor: float = 1) -> float:
    s_t = sigma * np.sqrt(tau)
    d = (forward_price - strike) / s_t
    return discount_factor * scst.norm.pdf(d) / s_t

def get_vanna(forward_price: float, strike: float, tau: float, sigma: float, discount_factor: float = 1) -> float:
    d = (forward_price - strike) / (sigma * np.sqrt(tau))
    return -discount_factor * scst.norm.pdf(d) * d / sigma * 1e-2

def get_volga(forward_price: float, strike: float, tau: float, sigma: float, discount_factor: float = 1) -> float:
    d = (forward_price - strike) / (sigma * np.sqrt(tau))
    return discount_factor * scst.norm.pdf(d) * d**2 * np.sqrt(tau) / sigma * 1e-4

def get_moneyness_for_delta(
        delta: float, tau: float, sigma: float, forward_price: float = None,
        moneyness_type = OptionMoneynessType.Simple) -> float:
    inv_n = scst.norm.ppf(abs(delta)) * (-1 if delta < 0 else 1)
    sigma_t = sigma * np.sqrt(tau)
    match moneyness_type:
        case OptionMoneynessType.Strike:
            return forward_price - sigma_t * inv_n
        case OptionMoneynessType.Simple:
            return -sigma_t * inv_n
        case OptionMoneynessType.Normal:
            return -sigma * inv_n
        case OptionMoneynessType.Standard:
            return -inv_n

def get_strike_for_delta(delta: float, forward_price: float, tau: float, sigma: float) -> float:
    return get_moneyness_for_delta(delta=delta, tau=tau, sigma=sigma, forward_price=forward_price,
            moneyness_type=OptionMoneynessType.Strike)

def get_moneyness(
        forward_price: float, strike: float, tau: float = None,
        sigma: float = None, moneyness_type = OptionMoneynessType.Simple) -> float:
    match moneyness_type:
        case OptionMoneynessType.Simple:
            return strike - forward_price
        case OptionMoneynessType.Normal:
            return (strike - forward_price) / np.sqrt(tau)
        case OptionMoneynessType.Standard:
            return (strike - forward_price) / (sigma * np.sqrt(tau))

def get_strike_for_moneyness(
        moneyness: float, forward_price: float, tau: float = None,
        sigma: float = None, moneyness_type = OptionMoneynessType.Simple) -> float:
    match moneyness_type:
        case OptionMoneynessType.Simple:
            return forward_price + moneyness
        case OptionMoneynessType.Normal:
            return forward_price + moneyness * np.sqrt(tau)
        case OptionMoneynessType.Standard:
            return forward_price + moneyness * sigma * np.sqrt(tau)

