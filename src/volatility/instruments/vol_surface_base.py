from pydantic.dataclasses import dataclass

from common.base_class import NameDateClass


@dataclass
class VolSurfaceBase(NameDateClass):
    
    def get_strike_vol(self, tau: float, strike: float, forward_price: float) -> float:
        '''Get volatility for tenor, strike and forward price'''
