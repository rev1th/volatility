import numpy as np
import scipy.stats as scst

from common.numeric import solver
from volatility.models.delta_types import FXDeltaType, OptionMoneynessType, ATMStrikeType
from volatility.lib import black_scholes

def get_delta(forward_price: float, strike: float, dcf: float, sigma: float, flag: int = 1,
                delta_type: FXDeltaType = FXDeltaType.Forward) -> float:
    d1, d2 = black_scholes.get_d12(forward_price, strike=strike, expiry_dcf=dcf, sigma=sigma)
    match delta_type:
        case FXDeltaType.Forward:
            return flag * scst.norm.cdf(flag * d1)
        case FXDeltaType.ForwardPremium:
            return flag * strike / forward_price * scst.norm.cdf(flag * d2)
        case _:
            raise Exception("FXDelta type not implemented yet")

def get_delta_for_moneyness(moneyness: float, dcf: float, sigma: float, flag: int = 1,
                            delta_type: FXDeltaType = FXDeltaType.Forward) -> float:
    lfk = -moneyness
    d1, d2 = black_scholes.get_d12_m(lfk, expiry_dcf=dcf, sigma=sigma)
    match delta_type:
        case FXDeltaType.Forward:
            return flag * scst.norm.cdf(flag * d1)
        case FXDeltaType.ForwardPremium:
            return flag * np.exp(-lfk) * scst.norm.cdf(flag * d2)
        case _:
            raise Exception("FXDelta type not implemented yet")

def get_moneyness_for_delta(
        delta: float, dcf: float, sigma: float, forward_price: float = None,
        delta_type: FXDeltaType = FXDeltaType.Forward,
        moneyness_type = OptionMoneynessType.LogSimple) -> float:
    match delta_type:
        case FXDeltaType.Forward:
            inv_n = scst.norm.ppf(abs(delta)) * (-1 if delta < 0 else 1)
            lkf = sigma * (sigma * dcf / 2 - inv_n * np.sqrt(dcf))
        case FXDeltaType.ForwardPremium:
            alpha = sigma * np.sqrt(dcf)
            root = np.log(2 * abs(delta)) + alpha**2 / 2
            d_sgn = -1 if delta < 0 else 1
            res = solver.find_root(
                error_f=lambda y: np.log(2 * scst.norm.cdf(-d_sgn * y)) + alpha * y - root,
                init_guess=0,
                f_prime=lambda y: alpha - d_sgn * scst.norm.pdf(-d_sgn*y) / scst.norm.cdf(-d_sgn*y),
            )
            lkf = alpha * res - alpha**2
        case _:
            raise Exception("FXDelta type not implemented yet")
    match moneyness_type:
        case OptionMoneynessType.Strike:
            return forward_price * np.exp(lkf)
        case OptionMoneynessType.Simple:
            return np.exp(lkf)
        case OptionMoneynessType.LogSimple:
            return lkf
        case OptionMoneynessType.Normal:
            return lkf / np.sqrt(dcf)
        case OptionMoneynessType.Standard:
            return lkf / (sigma * np.sqrt(dcf))

def get_strike_for_delta(delta: float, forward_price: float, dcf: float, sigma: float,
                            delta_type = FXDeltaType.Forward) -> float:
    return get_moneyness_for_delta(delta=delta, dcf=dcf, sigma=sigma, forward_price=forward_price,
                delta_type=delta_type, moneyness_type=OptionMoneynessType.Strike)

def get_moneyness_atm(atm_type: ATMStrikeType, dcf: float = None, sigma: float = None,
                        forward_price: float = None, delta_type = FXDeltaType.Forward,
                        moneyness_type = OptionMoneynessType.LogSimple) -> float:
    match atm_type:
        case ATMStrikeType.Forward:
            match moneyness_type:
                case OptionMoneynessType.Strike:
                    return forward_price
                case OptionMoneynessType.Simple:
                    return 1
                case _:
                    return 0
        case ATMStrikeType.DN:
            return get_moneyness_for_delta(delta=0.5, dcf=dcf, sigma=sigma, forward_price=forward_price,
                    delta_type=delta_type, moneyness_type=moneyness_type)

def get_delta_vol(delta: float, forward_price: float, strike: float, dcf: float,
                    delta_type = FXDeltaType.Forward) -> float:
    match delta_type:
        case FXDeltaType.Forward:
            inv_n = scst.norm.ppf(abs(delta)) * (-1 if delta < 0 else 1)
            return (inv_n + np.sqrt(inv_n**2 - 2*np.log(forward_price / strike))) / np.sqrt(dcf)
        case FXDeltaType.ForwardPremium:
            inv_n = scst.norm.ppf(abs(delta) * forward_price / strike) * (-1 if delta < 0 else 1)
            return (inv_n + np.sqrt(inv_n**2 + 2*np.log(forward_price / strike))) / np.sqrt(dcf)
        case _:
            raise Exception("FXDelta type not implemented yet")
