import numpy as np
import scipy.stats as scst

from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess

from volatility.models.construct_types import NumeraireConvention, OptionMoneynessType

# Standard log normal Black volatility
# https://github.com/vollib/lets_be_rational
def get_implied_volatility(
        option_price: float, forward_price: float, strike: float, tau: float,
        flag: int = 1, discount_factor: float = 1) -> float:
    fwd_premium = option_price / discount_factor
    return implied_volatility_from_a_transformed_rational_guess(
        fwd_premium, forward_price, strike, tau, flag)

# Black Scholes pricing for Eurpoean style options
def get_d12(forward_price: float, strike: float, tau: float, sigma: float) -> tuple[float, float]:
    return get_d12_m(np.log(forward_price / strike), tau=tau, sigma=sigma)

def get_d12_m(lfk: float, tau: float, sigma: float) -> tuple[float, float]:
    s_t = sigma * np.sqrt(tau)
    # risk-adjusted probability that the option will be exercised
    d1 = (lfk / s_t + s_t / 2)
    # probability of receiving the asset at expiration of the option
    d2 = (lfk / s_t - s_t / 2)
    return d1, d2

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
def get_premium(forward_price: float, strike: float, tau: float, sigma: float, flag: int = 1, 
                discount_factor: float = 1, numeraire = NumeraireConvention.Regular) -> float:
    d1, d2 = get_d12(forward_price=forward_price, strike=strike, tau=tau, sigma=sigma)
    premium = flag * (forward_price * scst.norm.cdf(flag * d1, loc=0, scale=1) - 
                   strike * scst.norm.cdf(flag * d2, loc=0, scale=1)) * discount_factor
    match numeraire:
        case NumeraireConvention.Regular:
            return premium
        case NumeraireConvention.Inverse:
            return premium / forward_price

def get_delta(forward_price: float, strike: float, tau: float, sigma: float,
              flag: int = 1, discount_factor: float = 1) -> float:
    d1, _ = get_d12(forward_price, strike, tau, sigma)
    return flag * discount_factor * scst.norm.cdf(flag * d1)

def get_vega(forward_price: float, strike: float, tau: float, sigma: float, 
             discount_factor: float = 1, numeraire = NumeraireConvention.Regular) -> float:
    d1, _ = get_d12(forward_price, strike, tau, sigma)
    vega = discount_factor * forward_price * scst.norm.pdf(d1) * np.sqrt(tau) * 1e-2
    match numeraire:
        case NumeraireConvention.Regular:
            return vega
        case NumeraireConvention.Inverse:
            return vega / forward_price

def get_theta(forward_price: float, strike: float, tau: float, sigma: float, flag: int = 1, 
              discount_factor: float = 1, tau_unit: float = 1/252, 
              numeraire = NumeraireConvention.Regular) -> float:
    d1, d2 = get_d12(forward_price, strike, tau, sigma)
    fwd_premium = flag * (forward_price * scst.norm.cdf(flag * d1) - strike * scst.norm.cdf(flag * d2))
    rate = -np.log(discount_factor) / tau
    theta = discount_factor * (-forward_price * scst.norm.pdf(d1) * sigma / (2 * np.sqrt(tau)) + 
                               rate * fwd_premium) * tau_unit
    match numeraire:
        case NumeraireConvention.Regular:
            return theta
        case NumeraireConvention.Inverse:
            return theta / forward_price

def get_gamma(forward_price: float, strike: float, tau: float, sigma: float, discount_factor: float = 1) -> float:
    d1, _ = get_d12(forward_price, strike, tau, sigma)
    return discount_factor * scst.norm.pdf(d1) / (forward_price * sigma * np.sqrt(tau))

def get_vanna(forward_price: float, strike: float, tau: float, sigma: float, discount_factor: float = 1) -> float:
    d1, d2 = get_d12(forward_price, strike, tau, sigma)
    return -discount_factor * scst.norm.pdf(d1) * d2 / sigma * 1e-2

def get_volga(forward_price: float, strike: float, tau: float, sigma: float, 
              discount_factor: float = 1, numeraire = NumeraireConvention.Regular) -> float:
    d1, d2 = get_d12(forward_price, strike, tau, sigma)
    volga = discount_factor * forward_price * scst.norm.pdf(d1) * np.sqrt(tau) * d1 * d2 / sigma * 1e-4
    match numeraire:
        case NumeraireConvention.Regular:
            return volga
        case NumeraireConvention.Inverse:
            return volga / forward_price

def get_pdf(forward_price: float, strike: float, tau: float, sigma: float) -> float:
    d1, _ = get_d12(forward_price, strike, tau, sigma)
    return scst.norm.pdf(d1) / (strike * sigma * np.sqrt(tau))

def get_moneyness_for_delta(
        delta: float, tau: float, sigma: float, forward_price: float = None,
        moneyness_type = OptionMoneynessType.LogSimple) -> float:
    inv_n = scst.norm.ppf(abs(delta)) * (-1 if delta < 0 else 1)
    sigma_t = sigma * np.sqrt(tau)
    match moneyness_type:
        case OptionMoneynessType.Strike:
            return forward_price * np.exp(sigma_t * (sigma_t / 2 - inv_n))
        case OptionMoneynessType.Simple:
            return np.exp(sigma_t * (sigma_t / 2 - inv_n))
        case OptionMoneynessType.LogSimple:
            return sigma_t * (sigma_t / 2 - inv_n)
        case OptionMoneynessType.Normal:
            return sigma * (sigma_t / 2 - inv_n)
        case OptionMoneynessType.Standard:
            return sigma_t / 2 - inv_n

def get_strike_for_delta(delta: float, forward_price: float, tau: float, sigma: float) -> float:
    return get_moneyness_for_delta(delta=delta, tau=tau, sigma=sigma, forward_price=forward_price,
            moneyness_type=OptionMoneynessType.Strike)

def get_moneyness(
        forward_price: float, strike: float, tau: float = None,
        sigma: float = None, moneyness_type = OptionMoneynessType.LogSimple) -> float:
    match moneyness_type:
        case OptionMoneynessType.Simple:
            return strike / forward_price
        case OptionMoneynessType.LogSimple:
            return np.log(strike / forward_price)
        case OptionMoneynessType.Normal:
            return np.log(strike / forward_price) / np.sqrt(tau)
        case OptionMoneynessType.Standard:
            return np.log(strike / forward_price) / (sigma * np.sqrt(tau))

def get_strike_for_moneyness(
        moneyness: float, forward_price: float, tau: float = None,
        sigma: float = None, moneyness_type = OptionMoneynessType.LogSimple) -> float:
    match moneyness_type:
        case OptionMoneynessType.Simple:
            return forward_price * moneyness
        case OptionMoneynessType.LogSimple:
            return forward_price * np.exp(moneyness)
        case OptionMoneynessType.Normal:
            return forward_price * np.exp(moneyness * np.sqrt(tau))
        case OptionMoneynessType.Standard:
            return forward_price * np.exp(moneyness * sigma * np.sqrt(tau))

