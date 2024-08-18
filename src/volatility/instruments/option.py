from pydantic.dataclasses import dataclass
from typing import Union
import datetime as dtm

from common.models.base_instrument import BaseInstrument
from common.chrono.daycount import DayCount

from .option_types import OptionType, ExerciseStyle
from .vol_surface import VolSurfaceBase
from volatility.lib import black_scholes


@dataclass
class Option(BaseInstrument):
    _underlying: BaseInstrument
    _expiry: dtm.date
    _strike: float

    _style = ExerciseStyle.European
    _daycount_type = DayCount.ACT365

    def __post_init__(self):
        if self.name is None:
            self.name = f"{self._underlying.name} {self.type} {self._strike}"
    
    @property
    def underlying(self):
        return self._underlying
    
    @property
    def expiry(self):
        return self._expiry
    
    @property
    def strike(self):
        return self._strike
    
    @property
    def flag(self) -> int:
        raise Exception('Abstract object function: flag')
    
    def get_underlying_price(self, date: dtm.date) -> float:
        return self._underlying.data[date]
    
    def get_expiry_dcf(self, date: dtm.date) -> float:
        return self._daycount_type.get_dcf(date, self._expiry)
    
    def get_intrinsic_value(self, date: dtm.date) -> float:
        return max(self.flag * (self.get_underlying_price(date) - self._strike), 0)
    
    def is_valid_price(self, date: dtm.date) -> bool:
        return date in self.data and self.data[date] and self.data[date] > self.get_intrinsic_value(date)
    
    def get_vol(self, vol_surface: VolSurfaceBase) -> float:
        return vol_surface.get_date_strike_vol(self._expiry, self._strike, self.get_underlying_price(vol_surface.date))
    
    def get_moneyness(self, date: dtm.date, moneyness_type) -> float:
        return black_scholes.get_moneyness(forward_price=self.get_underlying_price(date),
                strike=self._strike, dcf=self.get_expiry_dcf(date), moneyness_type=moneyness_type)
    
    def get_implied_vol(self, date: dtm.date, rate: float = 0) -> float:
        return black_scholes.get_implied_vol(
                option_price=self.data[date],
                forward_price=self.get_underlying_price(date),
                strike=self._strike,
                expiry_dcf=self.get_expiry_dcf(date),
                rate=rate,
                flag=self.flag)
    
    def get_price_for_vol(self, date: dtm.date, sigma: float, rate: float = 0) -> float:
        return black_scholes.get_price(
                forward_price=self.get_underlying_price(date),
                strike=self._strike,
                expiry_dcf=self.get_expiry_dcf(date),
                sigma=sigma,
                rate=rate,
                flag=self.flag)
    
    def get_price(self, vol_surface: VolSurfaceBase, rate: float = 0) -> float:
        return black_scholes.get_price(
                forward_price=self.get_underlying_price(vol_surface.date),
                strike=self._strike,
                expiry_dcf=self.get_expiry_dcf(vol_surface.date),
                sigma=self.get_vol(vol_surface),
                rate=rate,
                flag=self.flag)
    
    def get_delta(self, vol_surface: VolSurfaceBase) -> float:
        return black_scholes.get_delta(
                forward_price=self.get_underlying_price(vol_surface.date),
                strike=self._strike,
                dcf=self.get_expiry_dcf(vol_surface.date),
                sigma=self.get_vol(vol_surface),
                flag=self.flag)
    
    def get_gamma(self, vol_surface: VolSurfaceBase) -> float:
        return black_scholes.get_gamma(
                forward_price=self.get_underlying_price(vol_surface.date),
                strike=self._strike,
                dcf=self.get_expiry_dcf(vol_surface.date),
                sigma=self.get_vol(vol_surface))

@dataclass
class CallOption(Option):
    
    @property
    def type(self):
        return OptionType.Call
    
    @property
    def flag(self) -> int:
        return 1

@dataclass
class PutOption(Option):

    @property
    def type(self):
        return OptionType.Put

    @property
    def flag(self) -> int:
        return -1
