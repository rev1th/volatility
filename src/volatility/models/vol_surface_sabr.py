from pydantic.dataclasses import dataclass

from volatility.instruments.vol_surface_base import VolSurfaceBase
from volatility.lib import sabr
from .vol_surface import VolStrikeSlice

@dataclass
class SABRSurface(VolSurfaceBase):
    alpha: float
    beta: float
    rho: float
    volvol: float

    def get_strike_vol(self, tau: float, strike: float, forward_price: float) -> float:
        return sabr.get_vol(forward_price=forward_price, strike=strike, tau=tau,
                            alpha=self.alpha, volvol=self.volvol, beta=self.beta, rho=self.rho)

@dataclass
class SABRSlice(VolStrikeSlice):
    alpha: float
    beta: float
    rho: float
    volvol: float

    def get_strike_vol(self, tau: float, strike: float, forward_price: float) -> float:
        return sabr.get_vol(forward_price=forward_price, strike=strike, tau=tau,
                            alpha=self.alpha, volvol=self.volvol, beta=self.beta, rho=self.rho)
