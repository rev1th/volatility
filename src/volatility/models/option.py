
from pydantic.dataclasses import dataclass
import datetime as dtm

from common.models.base_instrument import BaseInstrumentP
from common.chrono import DayCount

from .option_types import OptionType, OptionStyle
from .vol_surface import VolSurfaceBase
from volatility.lib import option_analytics


@dataclass
class Option(BaseInstrumentP):
    _underlying: BaseInstrumentP
    _expiry: dtm.date
    _strike: float

    _style = OptionStyle.European
    _daycount_type = DayCount.ACT365

    def __post_init__(self):
        if self.name is None:
            self.name = f"{self._underlying.name} {self.type} {self._strike}"
    
    def set_market(self, date: dtm.date, price: float) -> None:
        super().set_market(date, price)
    
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
    
    def get_underlying_price(self) -> float:
        return self._underlying.price
    
    def get_expiry_dcf(self) -> float:
        return self._daycount_type.get_dcf(self.value_date, self._expiry)
    
    def get_intrinsic_value(self) -> float:
        return max(self.flag * (self.get_underlying_price() - self.strike), 0)
    
    def is_valid(self) -> bool:
        return self.price and self.get_underlying_price() and self.price > self.get_intrinsic_value()
    
    def get_vol(self, vol_surface: VolSurfaceBase) -> float:
        return vol_surface.get_vol(self.expiry, self.strike, self.get_underlying_price())
    
    def get_implied_vol(self, rate: float = 0) -> float:
        return option_analytics.get_implied_vol(
                option_price=self.price,
                forward_price=self.get_underlying_price(),
                strike=self.strike,
                expiry_dcf=self.get_expiry_dcf(),
                rate=rate,
                flag=self.flag)
    
    def get_price(self, vol_surface: VolSurfaceBase, rate: float = 0) -> float:
        return option_analytics.get_price(
                forward_price=self.get_underlying_price(),
                strike=self.strike,
                expiry_dcf=self.get_expiry_dcf(),
                sigma=self.get_vol(vol_surface),
                rate=rate,
                flag=self.flag)
    
    def get_delta(self, vol_surface: VolSurfaceBase) -> float:
        return option_analytics.get_delta(
                forward_price=self.get_underlying_price(),
                strike=self.strike,
                expiry_dcf=self.get_expiry_dcf(),
                sigma=self.get_vol(vol_surface),
                flag=self.flag)
    
    def get_gamma(self, vol_surface: VolSurfaceBase) -> float:
        return option_analytics.get_gamma(
                forward_price=self.get_underlying_price(),
                strike=self.strike,
                expiry_dcf=self.get_expiry_dcf(),
                sigma=self.get_vol(vol_surface),
            )

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
