from pydantic.dataclasses import dataclass
from dataclasses import field
import datetime as dtm

from common.base_class import NameDateClass
from common.chrono.daycount import DayCount


@dataclass
class VolSurfaceBase(NameDateClass):
    _daycount_type: DayCount = field(kw_only=True, default=DayCount.ACT365)

    def get_dcf(self, date: dtm.date) -> float:
        return self._daycount_type.get_dcf(self.date, date)
    
    def get_strike_vol(self, dcf: float, strike: float, forward_price: float) -> float:
        '''Get volatility for tenor, strike and forward price'''
    
    def get_date_strike_vol(self, date: dtm.date, strike: float, forward_price: float) -> float:
        return self.get_strike_vol(self.get_dcf(date), strike, forward_price)
