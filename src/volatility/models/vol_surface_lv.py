from pydantic.dataclasses import dataclass
import numpy as np

from volatility.models.vol_surface import VolSurfaceInterpolation
from volatility.lib import black_scholes


# https://medium.com/@add.mailme/implied-local-and-heston-volatility-and-its-calibration-in-python-1b3b05372af3
STRIKE_BUMP = 1e-3
TENOR_BUMP = 1e-2
@dataclass
class LocalVol(VolSurfaceInterpolation):
    _rate: float = 0
    
    def get_strike_vol(self, dcf: float, strike: float, forward_price: float) -> float:
        discount_factor = np.exp(-self._rate * dcf)
        d_strike = strike * STRIKE_BUMP
        strike_p1 = strike + d_strike
        strike_m1 = strike - d_strike
        price_k_p1 = black_scholes.get_premium(
                        forward_price=forward_price,
                        strike=strike_p1,
                        tau=dcf,
                        sigma=super().get_strike_vol(dcf, strike_p1, forward_price),
                        discount_factor=discount_factor,
                    )
        price_k = black_scholes.get_premium(
                        forward_price=forward_price,
                        strike=strike,
                        tau=dcf,
                        sigma=super().get_strike_vol(dcf, strike, forward_price),
                        discount_factor=discount_factor,
                    )
        price_k_m1 = black_scholes.get_premium(
                        forward_price=forward_price,
                        strike=strike_m1,
                        tau=dcf,
                        sigma=super().get_strike_vol(dcf, strike_m1, forward_price),
                        discount_factor=discount_factor,
                    )
        dprice_dkk = (price_k_p1 + price_k_m1 - 2 * price_k) / (d_strike ** 2)
        if dprice_dkk <= 0:
            return None

        d_dcf = dcf * TENOR_BUMP
        dcf_p1 = dcf + d_dcf
        dcf_m1 = dcf - d_dcf
        price_t_p1 = black_scholes.get_premium(
                        forward_price=forward_price,
                        strike=strike,
                        tau=dcf_p1,
                        sigma=super().get_strike_vol(dcf_p1, strike, forward_price),
                        discount_factor=discount_factor)
        price_t_m1 = black_scholes.get_premium(
                        forward_price=forward_price,
                        strike=strike,
                        tau=dcf_m1,
                        sigma=super().get_strike_vol(dcf_m1, strike, forward_price),
                        discount_factor=discount_factor)
        dprice_dt = (price_t_p1 - price_t_m1) / (d_dcf * 2)
        
        vol = np.sqrt((dprice_dt + self._rate * price_k) / (0.5 * strike**2 * dprice_dkk))
        return vol
