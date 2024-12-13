from pydantic.dataclasses import dataclass
from typing import ClassVar
import numpy as np
import bisect
import logging

from volatility.instruments.vol_surface_base import VolSurfaceBase
from volatility.models.delta_types import OptionMoneynessType
from volatility.lib.interpolator import Interpolator3D
from volatility.lib import black_scholes

logger = logging.Logger(__name__)

@dataclass
class VolSurfaceInterpolation(VolSurfaceBase):
    # (tau, moneyness, volatility)
    _slices: dict[float, list[tuple[float, float, float]]]
    _moneyness_type: OptionMoneynessType
    _interpolator_class = Interpolator3D.default()

    _interpolator: ClassVar[Interpolator3D]

    def __post_init__(self):
        self.set_interpolator()
    
    # interpolation of (tenor dcf, moneyness) -> variance
    def set_interpolator(self) -> None:
        surface_nodes = [(tau, ni[0], ni[1]**2 * tau) for tau, nodes in self._slices.items() for ni in nodes]
        self._interpolator = self._interpolator_class(surface_nodes)
    
    def get_vol(self, tau: float, moneyness: float) -> float:
        return np.sqrt(self._interpolator.get_value(tau, moneyness) / tau)
    
    def get_strike_vol(self, tau: float, strike: float, forward_price: float) -> float:
        moneyness = black_scholes.get_moneyness(strike=strike, forward_price=forward_price, tau=tau,
                                                moneyness_type=self._moneyness_type)
        return self.get_vol(tau, moneyness)


@dataclass
class VolStrikeSlice:
    _tau: float
    
    def get_strike_vol(self, tau: float, strike: float, forward_price: float) -> float:
        '''Get volatility for tenor, strike and forward price'''

@dataclass
class VolSurfaceSlices(VolSurfaceBase):
    _slice_curve: list[VolStrikeSlice]
    
    def get_strike_var(self, tau: float, strike: float, forward_price: float) -> float:
        s_i = bisect.bisect(self._slice_curve, tau, key=lambda sc: sc._tau)
        crv_i0 = self._slice_curve[s_i-1]
        tau_i0 = crv_i0._tau
        vol_i0 = crv_i0.get_strike_vol(tau=tau_i0, strike=strike, forward_price=forward_price)
        if s_i >= len(self._slice_curve) or tau <= tau_i0:
            return vol_i0**2 * tau
        crv_i1 = self._slice_curve[s_i]
        tau_i1 = crv_i1._tau
        vol_i1 = crv_i1.get_strike_vol(tau=tau_i1, strike=strike, forward_price=forward_price)
        weight = (tau - tau_i0) / (tau_i1 - tau_i0)
        # Monotonic variance for no-arbitrage
        return vol_i0**2 * tau_i0 * (1-weight) + vol_i1**2 * tau_i1 * weight
    
    def get_strike_vol(self, tau: float, strike: float, forward_price: float) -> float:
        return np.sqrt(self.get_strike_var(tau, strike, forward_price) / tau)

ROOT_EPS = 1e-6
@dataclass
class PolynomialMoneynessCurve(VolStrikeSlice):
    _xyw: list[tuple[float, float, float]]
    _moneyness_type: OptionMoneynessType
    _degree: int = 3

    def __post_init__(self):
        xs, ys, ws = zip(*self._xyw)
        assert(len(xs) > self._degree, f'Polynomial fit requires more than {self._degree} coordinates')
        self._polynomial = np.polynomial.Polynomial.fit(xs, ys, w=ws, deg=self._degree)
    
    def get_strike_vol(self, tau: float, strike: float, forward_price: float) -> float:
        if self._moneyness_type == OptionMoneynessType.Standard:
            moneyness, m_iter = 0, None
            while(not m_iter or abs(moneyness - m_iter) > ROOT_EPS):
                sigma = self._polynomial(moneyness)
                m_iter = moneyness
                moneyness = black_scholes.get_moneyness(strike=strike, forward_price=forward_price,
                                tau=tau, sigma=sigma, moneyness_type=OptionMoneynessType.Standard)
            return sigma
        moneyness = black_scholes.get_moneyness(strike=strike, forward_price=forward_price,
                        tau=tau, moneyness_type=self._moneyness_type)
        return self._polynomial(moneyness)
