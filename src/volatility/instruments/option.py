from pydantic.dataclasses import dataclass
import datetime as dtm

from common.models.base_instrument import BaseInstrument
from common.chrono.daycount import DayCount

from .option_types import OptionType, ExerciseStyle, OptionGreekType
from .vol_surface_base import VolSurfaceBase
from volatility.lib import black_scholes

PRICE_BUMP = 1e-4

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
    def expiry(self):
        return self._expiry
    
    @property
    def strike(self):
        return self._strike
    
    def get_forward_price(self, date: dtm.date) -> float:
        return self._underlying.data[date]
    
    def get_expiry_dcf(self, date: dtm.date) -> float:
        return self._daycount_type.get_dcf(date, self._expiry)
    
    def get_intrinsic_value(self, date: dtm.date) -> float:
        return max(self._flag * (self.get_forward_price(date) - self._strike), 0)
    
    def is_valid_price(self, date: dtm.date, discount_factor: float = 1) -> bool:
        return self.data.get(date, None) and self.data[date] > self.get_intrinsic_value(date) * discount_factor
    
    def get_volatility(self, vol_surface: VolSurfaceBase) -> float:
        date = vol_surface.date
        return vol_surface.get_strike_vol(self.get_expiry_dcf(date), self._strike, self.get_forward_price(date))
    
    def get_implied_volatility(self, date: dtm.date, discount_factor: float = 1) -> float:
        return black_scholes.get_implied_volatility(
                option_price=self.data[date],
                forward_price=self.get_forward_price(date),
                strike=self._strike,
                tau=self.get_expiry_dcf(date),
                discount_factor=discount_factor,
                flag=self._flag)
    
    def get_price_for_vol(self, date: dtm.date, sigma: float, discount_factor: float = 1) -> float:
        return black_scholes.get_premium(
                forward_price=self.get_forward_price(date),
                strike=self._strike,
                tau=self.get_expiry_dcf(date),
                sigma=sigma,
                discount_factor=discount_factor,
                flag=self._flag)
    
    def get_price(self, vol_surface: VolSurfaceBase, discount_factor: float = 1) -> float:
        date = vol_surface.date
        dcf, fwd_price = self.get_expiry_dcf(date), self.get_forward_price(date)
        return black_scholes.get_premium(
                forward_price=fwd_price,
                strike=self._strike,
                tau=dcf,
                sigma=vol_surface.get_strike_vol(dcf, self._strike, fwd_price),
                discount_factor=discount_factor,
                flag=self._flag)
    
    def get_delta(self, vol_surface: VolSurfaceBase) -> float:
        return self.get_greeks(vol_surface, [OptionGreekType.Delta_Black])[0]
    
    def get_vega(self, vol_surface: VolSurfaceBase) -> float:
        return self.get_greeks(vol_surface, [OptionGreekType.Vega_Black])[0]
    
    def get_gamma(self, vol_surface: VolSurfaceBase) -> float:
        return self.get_greeks(vol_surface, [OptionGreekType.Gamma_Black])[0]
    
    def get_greeks(self, vol_surface: VolSurfaceBase, greek_types: list[OptionGreekType],
                   discount_factor: float = 1) -> list[float]:
        forward_price = self.get_forward_price(vol_surface.date)
        expiry_dcf = self.get_expiry_dcf(vol_surface.date)
        strike, c_p_flag = self._strike, self._flag
        vol_F = vol_surface.get_strike_vol(expiry_dcf, strike, forward_price)
        if OptionGreekType.Delta_Adapted in greek_types or \
            OptionGreekType.Gamma_Adapted in greek_types:
            price_F = black_scholes.get_premium(
                forward_price=forward_price, strike=strike, tau=expiry_dcf,
                sigma=vol_F, flag=c_p_flag, discount_factor=discount_factor)
            dF = forward_price * PRICE_BUMP
            F_p = forward_price + dF
            F_m = forward_price - dF
            vol_F_p = vol_surface.get_strike_vol(expiry_dcf, strike, F_p)
            vol_F_m = vol_surface.get_strike_vol(expiry_dcf, strike, F_m)
            price_F_p = black_scholes.get_premium(
                forward_price=F_p, strike=strike, tau=expiry_dcf,
                sigma=vol_F_p, flag=c_p_flag, discount_factor=discount_factor)
            price_F_m = black_scholes.get_premium(
                forward_price=F_m, strike=strike, tau=expiry_dcf,
                sigma=vol_F_m, flag=c_p_flag, discount_factor=discount_factor)
        greeks = []
        for gtp in greek_types:
            match gtp:
                case OptionGreekType.Delta_Black:
                    measure = black_scholes.get_delta(
                        forward_price=forward_price, strike=strike,
                        tau=expiry_dcf, sigma=vol_F, flag=c_p_flag, discount_factor=discount_factor)
                case OptionGreekType.Gamma_Black:
                    measure = black_scholes.get_gamma(
                        forward_price=forward_price, strike=strike,
                        tau=expiry_dcf, sigma=vol_F, discount_factor=discount_factor)
                case OptionGreekType.Theta_Black:
                    measure = black_scholes.get_theta(
                        forward_price=forward_price, strike=strike,
                        tau=expiry_dcf, sigma=vol_F, flag=c_p_flag, discount_factor=discount_factor)
                case OptionGreekType.Vega_Black:
                    measure = black_scholes.get_vega(
                        forward_price=forward_price, strike=strike,
                        tau=expiry_dcf, sigma=vol_F, discount_factor=discount_factor)
                case OptionGreekType.Vanna_Black:
                    measure = black_scholes.get_vanna(
                        forward_price=forward_price, strike=strike,
                        tau=expiry_dcf, sigma=vol_F, discount_factor=discount_factor)
                case OptionGreekType.Volga_Black:
                    measure = black_scholes.get_volga(
                        forward_price=forward_price, strike=strike,
                        tau=expiry_dcf, sigma=vol_F, discount_factor=discount_factor)
                case OptionGreekType.Density:
                    measure = black_scholes.get_pdf(
                        forward_price=forward_price, strike=strike, tau=expiry_dcf, sigma=vol_F)
                case OptionGreekType.Delta_Adapted:
                    measure = (price_F_p - price_F_m) / (2 * dF)
                case OptionGreekType.Gamma_Adapted:
                    measure = (price_F_p + price_F_m - 2 * price_F) / (dF * dF)
            greeks.append(measure)
        return greeks

@dataclass
class CallOption(Option):

    def __post_init__(self):
        super().__post_init__()
        self._flag = 1
    
    @property
    def type(self):
        return OptionType.Call

@dataclass
class PutOption(Option):

    def __post_init__(self):
        super().__post_init__()
        self._flag = -1

    @property
    def type(self):
        return OptionType.Put
