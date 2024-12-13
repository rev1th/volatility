import numpy as np
import scipy.stats as scst

from common.numeric import solver
from volatility.models.delta_types import FXDeltaType, OptionMoneynessType, ATMStrikeType
from volatility.lib import black_scholes

def get_delta(forward_price: float, strike: float, tau: float, sigma: float, flag: int = 1,
                delta_type: FXDeltaType = FXDeltaType.Forward) -> float:
    d1, d2 = black_scholes.get_d12(forward_price, strike=strike, tau=tau, sigma=sigma)
    match delta_type:
        case FXDeltaType.Forward:
            return flag * scst.norm.cdf(flag * d1)
        case FXDeltaType.ForwardPremium:
            return flag * strike / forward_price * scst.norm.cdf(flag * d2)
        case _:
            raise Exception("FXDelta type not implemented yet")

def get_delta_for_moneyness(
        moneyness: float, tau: float, sigma: float, flag: int = 1,
        moneyness_type = OptionMoneynessType.LogSimple,
        delta_type: FXDeltaType = FXDeltaType.Forward) -> float:
    match moneyness_type:
        case OptionMoneynessType.Simple:
            log_f_k = -np.log(moneyness)
        case OptionMoneynessType.LogSimple:
            log_f_k = -moneyness
        case OptionMoneynessType.Normal:
            log_f_k = -moneyness * np.sqrt(tau)
        case OptionMoneynessType.Standard:
            log_f_k = -moneyness * sigma * np.sqrt(tau)
    d1, d2 = black_scholes.get_d12_m(log_f_k, tau=tau, sigma=sigma)
    match delta_type:
        case FXDeltaType.Forward:
            return flag * scst.norm.cdf(flag * d1)
        case FXDeltaType.ForwardPremium:
            return flag * np.exp(-log_f_k) * scst.norm.cdf(flag * d2)
        case _:
            raise Exception("FXDelta type not implemented yet")

def get_moneyness_for_delta(
        delta: float, tau: float, sigma: float, forward_price: float = None,
        delta_type: FXDeltaType = FXDeltaType.Forward,
        moneyness_type = OptionMoneynessType.LogSimple) -> float:
    match delta_type:
        case FXDeltaType.Forward:
            if delta is None:
                log_k_f = sigma**2 * tau / 2
            else:
                inv_n = scst.norm.ppf(abs(delta)) * (-1 if delta < 0 else 1)
                log_k_f = sigma * (sigma * tau / 2 - inv_n * np.sqrt(tau))
        case FXDeltaType.ForwardPremium:
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
        case _:
            raise Exception("FXDelta type not implemented yet")
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
        delta_type = FXDeltaType.Forward) -> float:
    return get_moneyness_for_delta(delta=delta, tau=tau, sigma=sigma, forward_price=forward_price,
                delta_type=delta_type, moneyness_type=OptionMoneynessType.Strike)

def get_moneyness_atm(
        atm_type: ATMStrikeType, tau: float = None, sigma: float = None, forward_price: float = None,
        delta_type = FXDeltaType.Forward, moneyness_type = OptionMoneynessType.LogSimple) -> float:
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
                    delta_type=delta_type, moneyness_type=moneyness_type)

def get_delta_atm(
        atm_type: ATMStrikeType, tau: float = None, sigma: float = None, flag: int = 1,
        delta_type = FXDeltaType.Forward) -> float:
    m_type = OptionMoneynessType.LogSimple
    moneyness = get_moneyness_atm(atm_type=atm_type, tau=tau, sigma=sigma,
                delta_type=delta_type, moneyness_type=m_type)
    return get_delta_for_moneyness(moneyness, tau=tau, sigma=sigma, flag=flag,
                delta_type=delta_type, moneyness_type=m_type)

def get_delta_complement(
        delta: float, tau: float = None, sigma: float = None, flag: int = 1,
        delta_type: FXDeltaType = FXDeltaType.Forward) -> float:
    assert delta * flag < 0, 'Require opposite signs to complement delta'
    match delta_type:
        case FXDeltaType.Forward:
            return flag + delta
        case FXDeltaType.ForwardPremium:
            k_f = get_moneyness_for_delta(delta, tau=tau, sigma=sigma,
                    delta_type=delta_type, moneyness_type=OptionMoneynessType.Simple)
            return flag * k_f + delta
        case _:
            raise Exception("FXDelta type not implemented yet")

def get_vol_for_delta(
        delta: float, forward_price: float, strike: float, tau: float,
        delta_type = FXDeltaType.Forward) -> float:
    d_sgn = (-1 if delta < 0 else 1)
    match delta_type:
        case FXDeltaType.Forward:
            inv_n = scst.norm.ppf(abs(delta))
            return (d_sgn * inv_n + np.sqrt(inv_n**2 - 2*np.log(forward_price / strike))) / np.sqrt(tau)
        case FXDeltaType.ForwardPremium:
            inv_n = scst.norm.ppf(abs(delta) * forward_price / strike)
            return (-d_sgn * inv_n + np.sqrt(inv_n**2 + 2*np.log(forward_price / strike))) / np.sqrt(tau)
        case _:
            raise Exception("FXDelta type not implemented yet")
