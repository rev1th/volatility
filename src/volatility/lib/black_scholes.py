import numpy as np
import scipy.stats as scst

from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess

from volatility.models.delta_types import OptionMoneynessType

# Standard log normal Black volatiity
# https://github.com/vollib/lets_be_rational
def get_implied_vol(option_price: float, forward_price: float, strike: float, expiry_dcf: float,
                    flag: int = 1, rate: float = 0) -> float:
    fwd_premium = option_price / np.exp(-rate * expiry_dcf)
    return implied_volatility_from_a_transformed_rational_guess(
        fwd_premium, forward_price, strike, expiry_dcf, flag)

# Black Scholes pricing for Eurpoean style options
def get_d12(forward_price: float, strike: float, expiry_dcf: float, sigma: float) -> tuple[float, float]:
    return get_d12_m(np.log(forward_price / strike), expiry_dcf=expiry_dcf, sigma=sigma)

def get_d12_m(lfk: float, expiry_dcf: float, sigma: float) -> tuple[float, float]:
    s_t = sigma * np.sqrt(expiry_dcf)
    # risk-adjusted probability that the option will be exercised
    d1 = (lfk / s_t + s_t / 2)
    # probability of receiving the asset at expiration of the option
    d2 = (lfk / s_t - s_t / 2)
    return d1, d2

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
def get_price(forward_price: float, strike: float, expiry_dcf: float, sigma: float,
                flag: int = 1, rate: float = 0) -> float:
    d1, d2 = get_d12(forward_price=forward_price, strike=strike, expiry_dcf=expiry_dcf, sigma=sigma)
    return flag * (forward_price * scst.norm.cdf(flag * d1, loc=0, scale=1) - 
                   strike * scst.norm.cdf(flag * d2, loc=0, scale=1)) * np.exp(-rate * expiry_dcf)

def get_delta(forward_price: float, strike: float, dcf: float, sigma: float, flag: int = 1) -> float:
    d1, _ = get_d12(forward_price, strike, dcf, sigma)
    return flag * scst.norm.cdf(flag * d1)

def get_gamma(forward_price: float, strike: float, dcf: float, sigma: float) -> float:
    d1, _ = get_d12(forward_price, strike, dcf, sigma)
    return scst.norm.pdf(d1) / (forward_price * sigma * np.sqrt(dcf))

def get_theta(forward_price: float, strike: float, dcf: float, sigma: float,
                flag: int = 1, rate: float = 0) -> float:
    d1, d2 = get_d12(forward_price, strike, dcf, sigma)
    return -forward_price * scst.norm.pdf(d1) * sigma / (2 * np.sqrt(dcf)) - \
        flag * rate * strike * np.exp(-rate * dcf) * scst.norm.cdf(flag * d2)

def get_vega(forward_price: float, strike: float, dcf: float, sigma: float) -> float:
    d1, _ = get_d12(forward_price, strike, dcf, sigma)
    return forward_price * scst.norm.pdf(d1) * np.sqrt(dcf)

def get_strike_for_delta(delta: float, forward_price: float, dcf: float, sigma: float) -> float:
    return forward_price * get_moneyness_for_delta(delta=delta, dcf=dcf, sigma=sigma,
                                                    moneyness_type=OptionMoneynessType.Simple)

def get_moneyness_for_delta(delta: float, dcf: float, sigma: float,
                            moneyness_type = OptionMoneynessType.Normal) -> float:
    inv_n = scst.norm.ppf(abs(delta)) * (-1 if delta < 0 else 1)
    sigma_t = sigma * np.sqrt(dcf)
    match moneyness_type:
        case OptionMoneynessType.Simple:
            return np.exp(sigma_t * (sigma_t / 2 - inv_n))
        case OptionMoneynessType.LogSimple:
            return sigma_t * (sigma_t / 2 - inv_n)
        case OptionMoneynessType.Normal:
            return sigma * (sigma_t / 2 - inv_n)
        case OptionMoneynessType.Standard:
            return sigma_t / 2 - inv_n

def get_moneyness(
        forward_price: float, strike: float, dcf: float = None,
        sigma: float = None, moneyness_type = OptionMoneynessType.Normal) -> float:
    match moneyness_type:
        case OptionMoneynessType.Simple:
            return strike / forward_price
        case OptionMoneynessType.LogSimple:
            return np.log(strike / forward_price)
        case OptionMoneynessType.Normal:
            return np.log(strike / forward_price) / np.sqrt(dcf)
        case OptionMoneynessType.Standard:
            return np.log(strike / forward_price) / (sigma * np.sqrt(dcf))

def get_strike_for_moneyness(
        moneyness: float, forward_price: float, dcf: float = None,
        sigma: float = None, moneyness_type = OptionMoneynessType.Normal) -> float:
    match moneyness_type:
        case OptionMoneynessType.Simple:
            return forward_price * moneyness
        case OptionMoneynessType.LogSimple:
            return forward_price * np.exp(moneyness)
        case OptionMoneynessType.Normal:
            return forward_price * np.exp(moneyness * np.sqrt(dcf))
        case OptionMoneynessType.Standard:
            return forward_price * np.exp(moneyness * sigma * np.sqrt(dcf))

