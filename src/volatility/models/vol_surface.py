from pydantic.dataclasses import dataclass
from typing import ClassVar
import datetime as dtm
import numpy as np
import bisect

from volatility.instruments.vol_surface import VolSurfaceBase
from volatility.models.delta_types import OptionMoneynessType
from volatility.lib.interpolator import Interpolator3D
from volatility.lib import black_scholes, sabr


@dataclass
class VolSurfaceInterpolation(VolSurfaceBase):
    # (expiry, moneyness, volatility)
    _nodes: list[tuple[dtm.date, float, float]]
    _moneyness_type: OptionMoneynessType
    _interpolator_class = Interpolator3D.default()

    _interpolator: ClassVar[Interpolator3D]

    def __post_init__(self):
        self.set_interpolator()
    
    # interpolation of (tenor dcf, moneyness) -> variance
    def set_interpolator(self) -> None:
        nodes = [(self.get_dcf(ni[0]), ni[1], ni[2]**2) for ni in self._nodes]
        self._interpolator = self._interpolator_class(nodes)
    
    def get_vol(self, dcf: float, moneyness: float) -> float:
        return np.sqrt(self._interpolator.get_value(dcf, moneyness))
    
    def get_strike_vol(self, dcf: float, strike: float, forward_price: float) -> float:
        moneyness = black_scholes.get_moneyness(strike=strike, forward_price=forward_price, dcf=dcf,
                                                moneyness_type=self._moneyness_type)
        return self.get_vol(dcf, moneyness)
    
    def get_error(self) -> float:
        errors = []
        for (dcf, moneyness, vol) in self._nodes:
            errors.append(self.get_vol(dcf, moneyness) - vol)
        return np.sqrt(np.mean(np.array(errors)**2))


@dataclass
class VolStrikeSlice:

    def get_value(self, _: float):
        '''Get volatility for strike slice'''
    
    def get_strike_vol(self, dcf: float, strike: float, forward_price: float) -> float:
        '''Get volatility for tenor, strike and forward price'''

@dataclass
class VolSurfaceSlices(VolSurfaceBase):
    _slice_curve: list[tuple[float, VolStrikeSlice]]
    
    def get_strike_vol(self, dcf: float, strike: float, forward_price: float) -> float:
        s_i = bisect.bisect(self._slice_curve, dcf, key=lambda s: s[0])
        t_i0, crv_i0 = self._slice_curve[s_i-1]
        vol_i0 = crv_i0.get_strike_vol(dcf=dcf, strike=strike, forward_price=forward_price)
        if s_i >= len(self._slice_curve):
            return vol_i0
        t_i1, crv_i1 = self._slice_curve[s_i]
        vol_i1 = crv_i1.get_strike_vol(dcf=dcf, strike=strike, forward_price=forward_price)
        t = (dcf - t_i0) / (t_i1 - t_i0)
        return np.sqrt(vol_i0**2 * (1-t) + vol_i1**2 * t)


# https://medium.com/@add.mailme/implied-local-and-heston-volatility-and-its-calibration-in-python-1b3b05372af3
STRIKE_BUMP = 1e-3
TENOR_BUMP = 1e-2
@dataclass
class LocalVol(VolSurfaceInterpolation):
    _rate: float = 0
    
    def get_strike_vol(self, dcf: float, strike: float, forward_price: float) -> float:
        d_strike = strike * STRIKE_BUMP
        strike_p1 = strike + d_strike
        strike_m1 = strike - d_strike
        price_k_p1 = black_scholes.get_price(
                        forward_price=forward_price,
                        strike=strike_p1,
                        expiry_dcf=dcf,
                        sigma=super().get_strike_vol(dcf, strike_p1, forward_price),
                        rate=self._rate,
                    )
        price_k = black_scholes.get_price(
                        forward_price=forward_price,
                        strike=strike,
                        expiry_dcf=dcf,
                        sigma=super().get_strike_vol(dcf, strike, forward_price),
                        rate=self._rate,
                    )
        price_k_m1 = black_scholes.get_price(
                        forward_price=forward_price,
                        strike=strike_m1,
                        expiry_dcf=dcf,
                        sigma=super().get_strike_vol(dcf, strike_m1, forward_price),
                        rate=self._rate,
                    )
        opt_dkk = (price_k_p1 + price_k_m1 - 2 * price_k) / (d_strike ** 2)
        if opt_dkk <= 0:
            return None

        d_dcf = dcf * TENOR_BUMP
        dcf_p1 = dcf + d_dcf
        dcf_m1 = dcf - d_dcf
        price_t_p1 = black_scholes.get_price(
                        forward_price=forward_price,
                        strike=strike,
                        expiry_dcf=dcf_p1,
                        sigma=super().get_strike_vol(dcf_p1, strike, forward_price),
                        rate=self._rate)
        price_t_m1 = black_scholes.get_price(
                        forward_price=forward_price,
                        strike=strike,
                        expiry_dcf=dcf_m1,
                        sigma=super().get_strike_vol(dcf_m1, strike, forward_price),
                        rate=self._rate)
        opt_dt = (price_t_p1 - price_t_m1) / (d_dcf * 2)
        
        vol = np.sqrt((opt_dt + self._rate * price_k) / (0.5 * strike**2 * opt_dkk))
        return vol

@dataclass
class SABR(VolSurfaceBase):
    alpha: float
    beta: float
    rho: float
    volvol: float

    def get_strike_vol(self, dcf: float, strike: float, forward_price: float) -> float:
        return sabr.get_vol(forward_price=forward_price, strike=strike, dcf=dcf,
                            alpha=self.alpha, volvol=self.volvol, beta=self.beta, rho=self.rho)
