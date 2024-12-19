from pydantic.dataclasses import dataclass
from dataclasses import field
import datetime as dtm

from common.models.base_instrument import BaseInstrument
from common.chrono.daycount import DayCount

from .option_types import ExerciseStyle
from .vol_surface_base import VolSurfaceBase
from volatility.lib import black_model


@dataclass
class Option(BaseInstrument):
    _underlying: BaseInstrument
    _expiry: dtm.datetime
    _strike: float

    _style: ExerciseStyle = field(kw_only=True, default=ExerciseStyle.European)
    _daycount_type: DayCount = field(kw_only=True, default=DayCount.BD252)

    def __post_init__(self):
        if self.name is None:
            self.name = f"{self._underlying.name} {self.type} {self._strike}"
    
    @property
    def expiry(self):
        return self._expiry
    
    @property
    def strike(self):
        return self._strike
    
    def get_forward_price(self, date: dtm.date) -> float:
        return self._underlying.data[date]
    
    def get_tau(self, date: dtm.date) -> float:
        return self._daycount_type.get_dcf(date, self._expiry.date(), self._calendar)
    
    def get_intrinsic_value(self, date: dtm.date) -> float:
        return max(self._flag * (self.get_forward_price(date) - self._strike), 0)
    
    def get_volatility(self, vol_surface: VolSurfaceBase) -> float:
        date = vol_surface.date
        return vol_surface.get_strike_vol(self.get_tau(date), self._strike, self.get_forward_price(date))
    
    def get_price(self, sigma: float, discount_factor: float = 1,
            date: dtm.date = None, forward_price: float = None, tau: float = None) -> float:
        if forward_price is None:
            forward_price = self.get_forward_price(date)
        if tau is None:
            tau = self.get_tau(date)
        return black_model.get_premium(
            forward_price=forward_price, strike=self._strike, tau=tau,
            sigma=sigma, discount_factor=discount_factor, flag=self._flag)
    
    def get_price_from_surface(self, vol_surface: VolSurfaceBase, discount_factor: float = 1) -> float:
        date = vol_surface.date
        tau, fwd_price = self.get_tau(date), self.get_forward_price(date)
        return black_model.get_premium(
                forward_price=fwd_price, strike=self._strike, tau=tau,
                sigma=vol_surface.get_strike_vol(tau, self._strike, fwd_price),
                discount_factor=discount_factor,
                flag=self._flag)
