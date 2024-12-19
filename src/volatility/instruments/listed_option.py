from pydantic.dataclasses import dataclass
import datetime as dtm

from .option import Option
from .option_types import OptionType, OptionGreekType
from .vol_surface_base import VolSurfaceBase
from volatility.lib import black_model

PRICE_BUMP = 1e-4

@dataclass
class ListedOption(Option):
    
    def is_valid_price(self, date: dtm.date, discount_factor: float = 1) -> bool:
        return self.data.get(date, None) and self.data[date] > self.get_intrinsic_value(date) * discount_factor
    
    def get_implied_volatility(self, date: dtm.date, discount_factor: float = 1) -> float:
        return black_model.get_implied_volatility(
                option_price=self.data[date],
                forward_price=self.get_forward_price(date),
                strike=self._strike,
                tau=self.get_tau(date),
                discount_factor=discount_factor,
                flag=self._flag)
    
    def get_delta(self, vol_surface: VolSurfaceBase) -> float:
        return self.get_greeks(vol_surface, [OptionGreekType.Delta_Black])[0]
    
    def get_vega(self, vol_surface: VolSurfaceBase) -> float:
        return self.get_greeks(vol_surface, [OptionGreekType.Vega_Black])[0]
    
    def get_gamma(self, vol_surface: VolSurfaceBase) -> float:
        return self.get_greeks(vol_surface, [OptionGreekType.Gamma_Black])[0]
    
    def get_greeks(self, vol_surface: VolSurfaceBase, greek_types: list[OptionGreekType],
                   discount_factor: float = 1) -> dict[OptionGreekType, float]:
        forward_price = self.get_forward_price(vol_surface.date)
        tau = self.get_tau(vol_surface.date)
        strike, c_p_flag = self._strike, self._flag
        vol_F = vol_surface.get_strike_vol(tau, strike, forward_price)
        if OptionGreekType.Delta_Adapted in greek_types or \
            OptionGreekType.Gamma_Adapted in greek_types:
            price_F = black_model.get_premium(
                forward_price=forward_price, strike=strike, tau=tau,
                sigma=vol_F, flag=c_p_flag, discount_factor=discount_factor)
            dF = forward_price * PRICE_BUMP
            F_p = forward_price + dF
            F_m = forward_price - dF
            vol_F_p = vol_surface.get_strike_vol(tau, strike, F_p)
            vol_F_m = vol_surface.get_strike_vol(tau, strike, F_m)
            price_F_p = black_model.get_premium(
                forward_price=F_p, strike=strike, tau=tau,
                sigma=vol_F_p, flag=c_p_flag, discount_factor=discount_factor)
            price_F_m = black_model.get_premium(
                forward_price=F_m, strike=strike, tau=tau,
                sigma=vol_F_m, flag=c_p_flag, discount_factor=discount_factor)
        greeks = {}
        for gtp in greek_types:
            match gtp:
                case OptionGreekType.Delta_Black:
                    measure = black_model.get_delta(
                        forward_price=forward_price, strike=strike, tau=tau, sigma=vol_F, 
                        flag=c_p_flag, discount_factor=discount_factor)
                case OptionGreekType.Gamma_Black:
                    measure = black_model.get_gamma(
                        forward_price=forward_price, strike=strike, tau=tau, sigma=vol_F, 
                        discount_factor=discount_factor)
                case OptionGreekType.Theta_Black:
                    tau_unit = self._daycount_type.get_unit_dcf()
                    measure = black_model.get_theta(
                        forward_price=forward_price, strike=strike, tau=tau, sigma=vol_F, 
                        flag=c_p_flag, discount_factor=discount_factor, tau_unit=tau_unit)
                case OptionGreekType.Vega_Black:
                    measure = black_model.get_vega(
                        forward_price=forward_price, strike=strike, tau=tau, sigma=vol_F, 
                        discount_factor=discount_factor)
                case OptionGreekType.Vanna_Black:
                    measure = black_model.get_vanna(
                        forward_price=forward_price, strike=strike, tau=tau, sigma=vol_F, 
                        discount_factor=discount_factor)
                case OptionGreekType.Volga_Black:
                    measure = black_model.get_volga(
                        forward_price=forward_price, strike=strike, tau=tau, sigma=vol_F, 
                        discount_factor=discount_factor)
                case OptionGreekType.Density:
                    measure = black_model.get_pdf(
                        forward_price=forward_price, strike=strike, tau=tau, sigma=vol_F)
                case OptionGreekType.Delta_Adapted:
                    measure = (price_F_p - price_F_m) / (2 * dF)
                case OptionGreekType.Gamma_Adapted:
                    measure = (price_F_p + price_F_m - 2 * price_F) / (dF * dF)
            greeks[gtp] = measure
        return greeks

@dataclass
class CallOption(ListedOption):

    def __post_init__(self):
        self.type = OptionType.Call
        super().__post_init__()
        self._flag = 1

@dataclass
class PutOption(ListedOption):

    def __post_init__(self):
        self.type = OptionType.Put
        super().__post_init__()
        self._flag = -1
