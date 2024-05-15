
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import ClassVar
from abc import abstractmethod
import datetime as dtm
import numpy as np

from common.base_class import NameDateClass
from common.chrono import DayCount

from volatility.lib.interpolator import Interpolator3D
from volatility.lib import option_analytics


# Abstract class
@dataclass
class VolSurfaceBase(NameDateClass):
    _daycount_type: DayCount = field(kw_only=True, default=DayCount.ACT365)

    def get_dcf(self, date: dtm.date) -> float:
        return self._daycount_type.get_dcf(self.date, date)

    @abstractmethod
    def get_tenor_vol(self, dcf: float, strike: float, forward_price: float = None) -> float:
        '''Get volatility for tenor, strike and forward price'''
    
    def get_vol(self, date: dtm.date, strike: float, forward_price: float = None) -> float:
        return self.get_tenor_vol(self.get_dcf(date), strike, forward_price)

@dataclass
class VolSurfaceInterp(VolSurfaceBase):
    _nodes: list[tuple[dtm.date, float, float]]
    _interpolator_class = Interpolator3D.default()

    _interpolator: ClassVar[Interpolator3D]

    def __post_init__(self):
        self.set_interpolator()
    
    # interpolation of (tenor in dcf, strike) vs variance
    def set_interpolator(self) -> None:
        nodes = [(self.get_dcf(ni[0]), ni[1], ni[2]**2) for ni in self._nodes]
        self._interpolator = self._interpolator_class(nodes)
    
    @property
    def nodes(self):
        return self._nodes
    
    def get_node_keys(self):
        return dict.fromkeys([(ni[0], ni[1]) for ni in self._nodes]).keys()
    
    def set_node(self, date: dtm.date, strike: float, vol: float) -> None:
        for i, nd in enumerate(self._nodes):
            if nd[0] == date and nd[1] == strike:
                self._nodes[i] = (date, strike, vol)
                self.set_interpolator()
                return
    
    def get_tenor_vol(self, dcf: float, strike: float, _: float = None) -> float:
        return np.sqrt(self._interpolator.get_value(dcf, strike))
    
    def get_error(self) -> float:
        errors = []
        for (date, strike, vol) in self._nodes:
            errors.append(self.get_vol(date, strike) - vol)
        return np.sqrt(np.mean(np.array(errors)**2))


# https://medium.com/@add.mailme/implied-local-and-heston-volatility-and-its-calibration-in-python-1b3b05372af3
STRIKE_BUMP = 1e-3
TENOR_BUMP = 1e-2
@dataclass
class LV(VolSurfaceInterp):
    _rate: float = 0
    
    def get_tenor_vol(self, dcf: float, strike: float, forward_price: float) -> float:
        d_strike = strike * STRIKE_BUMP
        strike_p1 = strike + d_strike
        strike_m1 = strike - d_strike
        price_k_p1 = option_analytics.get_price(
                        forward_price=forward_price,
                        strike=strike_p1,
                        expiry_dcf=dcf,
                        sigma=super().get_tenor_vol(dcf, strike_p1),
                        rate=self._rate,
                    )
        price_k = option_analytics.get_price(
                        forward_price=forward_price,
                        strike=strike,
                        expiry_dcf=dcf,
                        sigma=super().get_tenor_vol(dcf, strike),
                        rate=self._rate,
                    )
        price_k_m1 = option_analytics.get_price(
                        forward_price=forward_price,
                        strike=strike_m1,
                        expiry_dcf=dcf,
                        sigma=super().get_tenor_vol(dcf, strike_m1),
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
                        sigma=super().get_tenor_vol(dcf_p1, strike),
                        rate=self._rate)
        price_t_m1 = option_analytics.get_price(
                        forward_price=forward_price,
                        strike=strike,
                        expiry_dcf=dcf_m1,
                        sigma=super().get_tenor_vol(dcf_m1, strike),
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

    def get_tenor_vol(self, dcf: float, strike: float, forward_price: float) -> float:
        return option_analytics.get_SABR_vol(forward_price=forward_price, strike=strike, dcf=dcf,
                                      alpha=self.alpha, volvol=self.volvol, beta=self.beta, rho=self.rho)
