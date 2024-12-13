from pydantic.dataclasses import dataclass

from common.base_class import NameDateClass
from .vol_types import VolatilityModelType
from .vol_surface import VolSurfaceBase

@dataclass
class VolSurfaceConstruct(NameDateClass):
    
    def get_implied(self):
        '''Get implied volatility from market quotes'''
    
    def build(self, _: VolatilityModelType = None) -> VolSurfaceBase:
        '''Build calibration for different models'''
