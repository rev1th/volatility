from pydantic.dataclasses import dataclass
import numpy as np

from .vol_surface import VolSurfaceInterpolation
from volatility.lib import black_model


# https://medium.com/@add.mailme/implied-local-and-heston-volatility-and-its-calibration-in-python-1b3b05372af3
STRIKE_BUMP = 1e-3
TENOR_BUMP = 1e-2
@dataclass
class LocalVolatility(VolSurfaceInterpolation):
    _discount_curve: dict[float, float]

    def get_rate(self, tau: float):
        if not self._discount_curve:
            return 1, 0
        tau_0 = None
        for tau_1, df_1 in self._discount_curve.items():
            if tau_1 > tau:
                if tau_0 is None:
                    break
                wt = (tau - tau_0) / (tau_1 - tau_0)
                return df_0**(1-wt) * df_1**wt, -(np.log(df_0) * (1-wt) + np.log(df_1) * wt) / tau
            tau_0, df_0 = tau_1, df_1
        return df_1**(tau / tau_1), -np.log(df_1) / tau_1
    
    def get_strike_vol(self, tau: float, strike: float, forward_price: float) -> float:
        discount_factor, rate = self.get_rate(tau)
        d_strike = strike * STRIKE_BUMP
        strike_p = strike + d_strike
        strike_m = strike - d_strike
        price_k_p = black_model.get_premium(
                        forward_price=forward_price,
                        strike=strike_p,
                        tau=tau,
                        sigma=super().get_strike_vol(tau, strike_p, forward_price),
                        discount_factor=discount_factor,
                    )
        price_k = black_model.get_premium(
                        forward_price=forward_price,
                        strike=strike,
                        tau=tau,
                        sigma=super().get_strike_vol(tau, strike, forward_price),
                        discount_factor=discount_factor,
                    )
        price_k_m = black_model.get_premium(
                        forward_price=forward_price,
                        strike=strike_m,
                        tau=tau,
                        sigma=super().get_strike_vol(tau, strike_m, forward_price),
                        discount_factor=discount_factor,
                    )
        dprice_dkk = (price_k_p + price_k_m - 2 * price_k) / (d_strike ** 2)
        if dprice_dkk <= 0:
            return None

        d_tau = tau * TENOR_BUMP
        tau_p = tau + d_tau
        tau_m = tau - d_tau
        price_t_p = black_model.get_premium(
                        forward_price=forward_price,
                        strike=strike,
                        tau=tau_p,
                        sigma=super().get_strike_vol(tau_p, strike, forward_price),
                        discount_factor=discount_factor)
        price_t_m = black_model.get_premium(
                        forward_price=forward_price,
                        strike=strike,
                        tau=tau_m,
                        sigma=super().get_strike_vol(tau_m, strike, forward_price),
                        discount_factor=discount_factor)
        dprice_dt = (price_t_p - price_t_m) / (d_tau * 2)
        
        var = (dprice_dt + rate * price_k) / (0.5 * strike**2 * dprice_dkk)
        return np.sqrt(var)
