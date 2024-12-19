import numpy as np
import scipy.stats as scst

from common.numeric import solver
from volatility.models.construct_types import NumeraireConvention, OptionMoneynessType, ATMStrikeType
from volatility.lib import black_model

def get_delta(forward_price: float, strike: float, tau: float, sigma: float, flag: int = 1,
        discount_factor: float = 1, numeraire = NumeraireConvention.Regular) -> float:
    d1, d2 = black_model.get_d12(forward_price, strike=strike, tau=tau, sigma=sigma)
    match numeraire:
        case NumeraireConvention.Regular:
            return flag * discount_factor * scst.norm.cdf(flag * d1)
        case NumeraireConvention.Inverse:
            return flag * discount_factor * strike / forward_price * scst.norm.cdf(flag * d2)

def get_gamma(forward_price: float, strike: float, tau: float, sigma: float, flag: int = 1,
        discount_factor: float = 1, numeraire = NumeraireConvention.Regular) -> float:
    match numeraire:
        case NumeraireConvention.Regular:
            return black_model.get_gamma(forward_price, strike=strike, tau=tau, sigma=sigma,
                                           discount_factor=discount_factor)
        case NumeraireConvention.Inverse:
            d1, d2 = black_model.get_d12(forward_price, strike, tau, sigma)
            return discount_factor * (scst.norm.pdf(d1) / (forward_price * sigma * np.sqrt(tau)) -
                flag * strike / forward_price**2 * scst.norm.cdf(flag * d2))

def get_vanna(forward_price: float, strike: float, tau: float, sigma: float,
        discount_factor: float = 1, numeraire = NumeraireConvention.Regular) -> float:
    match numeraire:
        case NumeraireConvention.Regular:
            return black_model.get_vanna(forward_price=forward_price, strike=strike, tau=tau, sigma=sigma,
                                           discount_factor=discount_factor)
        case NumeraireConvention.Inverse:
            d1, d2 = black_model.get_d12(forward_price, strike=strike, tau=tau, sigma=sigma)
            return -discount_factor * strike / forward_price * scst.norm.pdf(d2) * d1 / sigma * 1e-2

def get_delta_for_moneyness(
        moneyness: float, tau: float, sigma: float, flag: int = 1,
        moneyness_type = OptionMoneynessType.LogSimple,
        numeraire = NumeraireConvention.Regular) -> float:
    match moneyness_type:
        case OptionMoneynessType.Simple:
            log_f_k = -np.log(moneyness)
        case OptionMoneynessType.LogSimple:
            log_f_k = -moneyness
        case OptionMoneynessType.Normal:
            log_f_k = -moneyness * np.sqrt(tau)
        case OptionMoneynessType.Standard:
            log_f_k = -moneyness * sigma * np.sqrt(tau)
    d1, d2 = black_model.get_d12_m(log_f_k, tau=tau, sigma=sigma)
    match numeraire:
        case NumeraireConvention.Regular:
            return flag * scst.norm.cdf(flag * d1)
        case NumeraireConvention.Inverse:
            return flag * np.exp(-log_f_k) * scst.norm.cdf(flag * d2)

def get_moneyness_for_delta(
        delta: float, tau: float, sigma: float, forward_price: float = None,
        numeraire = NumeraireConvention.Regular,
        moneyness_type = OptionMoneynessType.LogSimple) -> float:
    match numeraire:
        case NumeraireConvention.Regular:
            if delta is None:
                log_k_f = sigma**2 * tau / 2
            else:
                inv_n = scst.norm.ppf(abs(delta)) * (-1 if delta < 0 else 1)
                log_k_f = sigma * (sigma * tau / 2 - inv_n * np.sqrt(tau))
        case NumeraireConvention.Inverse:
            if delta is None:
                log_k_f = -sigma**2 * tau / 2
            else:
                alpha = sigma * np.sqrt(tau)
                root = np.log(2 * abs(delta)) + alpha**2 / 2
                d_sgn = 1 if delta < 0 else -1 # flip sign for convenience
                res = solver.find_root(
                    error_f=lambda y: np.log(2 * scst.norm.cdf(d_sgn * y)) + alpha * y - root,
                    init_guess=0,
                    f_prime=lambda y: d_sgn * scst.norm.pdf(d_sgn*y) / scst.norm.cdf(d_sgn*y) + alpha,
                )
                log_k_f = alpha * res - alpha**2 / 2
    match moneyness_type:
        case OptionMoneynessType.Strike:
            return forward_price * np.exp(log_k_f)
        case OptionMoneynessType.Simple:
            return np.exp(log_k_f)
        case OptionMoneynessType.LogSimple:
            return log_k_f
        case OptionMoneynessType.Normal:
            return log_k_f / np.sqrt(tau)
        case OptionMoneynessType.Standard:
            return log_k_f / (sigma * np.sqrt(tau))

def get_strike_for_delta(
        delta: float, forward_price: float, tau: float, sigma: float,
        numeraire = NumeraireConvention.Regular) -> float:
    return get_moneyness_for_delta(delta=delta, tau=tau, sigma=sigma, forward_price=forward_price,
                numeraire=numeraire, moneyness_type=OptionMoneynessType.Strike)

def get_moneyness_atm(
        atm_type: ATMStrikeType, tau: float = None, sigma: float = None, forward_price: float = None,
        numeraire = NumeraireConvention.Regular, moneyness_type = OptionMoneynessType.LogSimple) -> float:
    match atm_type:
        case ATMStrikeType.Forward:
            match moneyness_type:
                case OptionMoneynessType.Strike:
                    return forward_price
                case OptionMoneynessType.Simple:
                    return 1
                case _:
                    return 0
        case ATMStrikeType.DeltaNeutral:
            return get_moneyness_for_delta(delta=None, tau=tau, sigma=sigma, forward_price=forward_price,
                    numeraire=numeraire, moneyness_type=moneyness_type)

def get_delta_atm(
        atm_type: ATMStrikeType, tau: float = None, sigma: float = None, flag: int = 1,
        numeraire = NumeraireConvention.Regular) -> float:
    m_type = OptionMoneynessType.LogSimple
    moneyness = get_moneyness_atm(atm_type=atm_type, tau=tau, sigma=sigma,
                numeraire=numeraire, moneyness_type=m_type)
    return get_delta_for_moneyness(moneyness, tau=tau, sigma=sigma, flag=flag,
                numeraire=numeraire, moneyness_type=m_type)

def get_delta_complement(
        delta: float, tau: float = None, sigma: float = None, flag: int = 1,
        numeraire = NumeraireConvention.Regular) -> float:
    assert delta * flag < 0, 'Require opposite signs to complement delta'
    match numeraire:
        case NumeraireConvention.Regular:
            return flag + delta
        case NumeraireConvention.Inverse:
            k_f = get_moneyness_for_delta(delta, tau=tau, sigma=sigma,
                    numeraire=numeraire, moneyness_type=OptionMoneynessType.Simple)
            return flag * k_f + delta

def get_vol_for_delta(
        delta: float, forward_price: float, strike: float, tau: float,
        numeraire = NumeraireConvention.Regular) -> float:
    d_sgn = (-1 if delta < 0 else 1)
    match numeraire:
        case NumeraireConvention.Regular:
            inv_n = scst.norm.ppf(abs(delta))
            root = inv_n**2 - 2*np.log(forward_price / strike)
            if root < 0:
                raise RuntimeError(f'Invalid delta {delta} for {strike} to calculate vol')
            return (d_sgn * inv_n + np.sqrt(root)) / np.sqrt(tau)
        case NumeraireConvention.Inverse:
            inv_n = scst.norm.ppf(abs(delta) * forward_price / strike)
            root = inv_n**2 + 2*np.log(forward_price / strike)
            if root < 0:
                raise RuntimeError(f'Invalid delta {delta} for {strike} to calculate vol')
            return (-d_sgn * inv_n + np.sqrt(root)) / np.sqrt(tau)
