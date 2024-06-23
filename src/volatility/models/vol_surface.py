
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import ClassVar
import datetime as dtm
import numpy as np
import bisect

from common.base_class import NameDateClass
from common.chrono import DayCount
from common.interpolator import Interpolator

from volatility.lib.interpolator import Interpolator3D
from volatility.lib import option_analytics


# Abstract class
@dataclass
class VolSurfaceBase(NameDateClass):
    _daycount_type: DayCount = field(kw_only=True, default=DayCount.ACT365)

    def get_dcf(self, date: dtm.date) -> float:
        return self._daycount_type.get_dcf(self.date, date)
    
    def get_strike_vol(self, dcf: float, strike: float, forward_price: float) -> float:
        '''Get volatility for tenor, strike and forward price'''
    
    def get_date_strike_vol(self, date: dtm.date, strike: float, forward_price: float) -> float:
        return self.get_strike_vol(self.get_dcf(date), strike, forward_price)

@dataclass
class VolSurfaceInterp(VolSurfaceBase):
    # (date, moneyness, volatility)
    _nodes: list[tuple[dtm.date, float, float]]
    _interpolator_class = Interpolator3D.default()

    _interpolator: ClassVar[Interpolator3D]

    def __post_init__(self):
        self.set_interpolator()
    
    # interpolation of (tenor dcf, moneyness) vs variance
    def set_interpolator(self) -> None:
        nodes = [(self.get_dcf(ni[0]), ni[1], ni[2]**2) for ni in self._nodes]
        self._interpolator = self._interpolator_class(nodes)
    
    @property
    def nodes(self):
        return self._nodes
    
    def get_node_keys(self):
        return dict.fromkeys([(ni[0], ni[1]) for ni in self._nodes]).keys()
    
    def get_vol(self, dcf: float, moneyness: float) -> float:
        return np.sqrt(self._interpolator.get_value(dcf, moneyness))
    
    def get_strike_vol(self, dcf: float, strike: float, price: float) -> float:
        return self.get_vol(dcf, option_analytics.get_moneyness(strike, price, dcf))
    
    def get_error(self) -> float:
        errors = []
        for (dcf, moneyness, vol) in self._nodes:
            errors.append(self.get_vol(dcf, moneyness) - vol)
        return np.sqrt(np.mean(np.array(errors)**2))

@dataclass
class VolSurfaceSlices(VolSurfaceBase):
    _slice_curve: ClassVar[list[tuple[float, Interpolator]]]
    
    def get_vol(self, dcf: float, moneyness: float) -> float:
        s_i = bisect.bisect(self._slice_curve, dcf)
        t_i0, crv_i0 = self._slice_curve[s_i-1]
        t_i1, crv_i1 = self._slice_curve[s_i]
        var_i0 = crv_i0.get_value(moneyness)**2
        var_i1 = crv_i1.get_value(moneyness)**2
        t = (dcf - t_i0) / (t_i1 - t_i0)
        return np.sqrt(var_i0 * (1 -t) + var_i1 * t)
    
    def get_strike_vol(self, dcf: float, strike: float, price: float) -> float:
        return self.get_vol(dcf, option_analytics.get_moneyness(strike, price, dcf))


# https://medium.com/@add.mailme/implied-local-and-heston-volatility-and-its-calibration-in-python-1b3b05372af3
STRIKE_BUMP = 1e-3
TENOR_BUMP = 1e-2
@dataclass
class LV(VolSurfaceInterp):
    _rate: float = 0
    
    def get_strike_vol(self, dcf: float, strike: float, forward_price: float) -> float:
        d_strike = strike * STRIKE_BUMP
        strike_p1 = strike + d_strike
        strike_m1 = strike - d_strike
        price_k_p1 = option_analytics.get_price(
                        forward_price=forward_price,
                        strike=strike_p1,
                        expiry_dcf=dcf,
                        sigma=super().get_strike_vol(dcf, strike_p1, forward_price),
                        rate=self._rate,
                    )
        price_k = option_analytics.get_price(
                        forward_price=forward_price,
                        strike=strike,
                        expiry_dcf=dcf,
                        sigma=super().get_strike_vol(dcf, strike, forward_price),
                        rate=self._rate,
                    )
        price_k_m1 = option_analytics.get_price(
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
        price_t_p1 = option_analytics.get_price(
                        forward_price=forward_price,
                        strike=strike,
                        expiry_dcf=dcf_p1,
                        sigma=super().get_strike_vol(dcf_p1, strike, forward_price),
                        rate=self._rate)
        price_t_m1 = option_analytics.get_price(
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
        return option_analytics.get_SABR_vol(forward_price=forward_price, strike=strike, dcf=dcf,
                                      alpha=self.alpha, volvol=self.volvol, beta=self.beta, rho=self.rho)
