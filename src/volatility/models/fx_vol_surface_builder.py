from pydantic.dataclasses import dataclass
import datetime as dtm
import numpy as np
from scipy import optimize

from common.numeric.interpolator import LogCubicSplineNatural
from .option_types import FXDeltaType, ATMStrikeType, OptionMoneynessType
from .vol_surface_builder import VolSurfaceModel
from .vol_surface import VolSurfaceSlices, VolSlice
from .vol_types import VolatilityQuoteType, VolSurfaceType
from volatility.lib import fx_delta, sabr, black_scholes


@dataclass
class FXVolQuote:
    _type: VolatilityQuoteType
    _value: float
    _delta: float | None = None

@dataclass
class PolynomialCurve(VolSlice):
    _coeffs: list[float]
    _delta_atm: float
    _delta_type: FXDeltaType

    def get_value(self, delta: float):
        dx = delta - self._delta_atm
        y = 0
        for i, v in enumerate(self._coeffs):
            y += v * dx**i
        return y
    
    def get_strike_vol(self, dcf: float, strike: float, forward_price: float):
        vol_atm = self.get_value(0.5)
        delta = fx_delta.get_delta(strike=strike, forward_price=forward_price, dcf=dcf,
                                    sigma=vol_atm, delta_type=self._delta_type)
        return self.get_value(delta)

@dataclass
class SplineCurve(LogCubicSplineNatural, VolSlice):
    _extrapolate_left: bool = True
    
    def get_strike_vol(self, dcf: float, strike: float, forward_price: float) -> float:
        vol_atm = self.get_value(0)
        moneyness = black_scholes.get_moneyness(strike=strike, forward_price=forward_price,
                        dcf=dcf, sigma=vol_atm, moneyness_type=OptionMoneynessType.Standard)
        return self.get_value(moneyness)

@dataclass
class FXVolSurfaceModel(VolSurfaceModel):
    # (forward_expiry, forward_price): (delta, vol_quote)
    _vol_quotes: dict[tuple[dtm.date, float], list[FXVolQuote]]
    _delta_type: FXDeltaType = FXDeltaType.ForwardPremium

    def get_dcf(self, date: dtm.date) -> float:
        return (date - self.date).days / 365
    
    def build_implied(self, moneyness_type = OptionMoneynessType.LogSimple):
        vols_implied = {}
        for (expiry, f_price), vol_quotes in self._vol_quotes.items():
            expiry_dcf = self.get_dcf(expiry)
            bf_vols: dict[float, float] = {}
            rr_vols: dict[float, float] = {}
            moneyness_vols: list[tuple[float, float]] = []
            for quote in vol_quotes:
                match quote._type:
                    case VolatilityQuoteType.ATM:
                        vol_atm = quote._value
                    case VolatilityQuoteType.RiskReversal:
                        rr_vols[quote._delta] = quote._value
                    case VolatilityQuoteType.Butterfly:
                        bf_vols[quote._delta] = quote._value
            atm_m = fx_delta.get_moneyness_atm(atm_type=ATMStrikeType.DN, dcf=expiry_dcf, sigma=vol_atm,
                            forward_price=f_price, delta_type=self._delta_type, moneyness_type=moneyness_type)
            moneyness_vols.append((atm_m, vol_atm))
            for delta in bf_vols:
                put_vol = vol_atm + bf_vols[delta] - rr_vols[delta] / 2
                put_m = fx_delta.get_moneyness_for_delta(-delta, dcf=expiry_dcf, sigma=put_vol,
                            forward_price=f_price, delta_type=self._delta_type, moneyness_type=moneyness_type)
                moneyness_vols.append((put_m, put_vol))
                call_vol = vol_atm + bf_vols[delta] + rr_vols[delta] / 2
                call_m = fx_delta.get_moneyness_for_delta(delta, dcf=expiry_dcf, sigma=call_vol,
                            forward_price=f_price, delta_type=self._delta_type, moneyness_type=moneyness_type)
                moneyness_vols.append((call_m, call_vol))
            vols_implied[(expiry, f_price)] = moneyness_vols
        return vols_implied
    
    def build_SM(self):
        implied_vols = self.build_implied(OptionMoneynessType.Standard)
        slice_curves = []
        for (expiry, _), moneyness_vols in implied_vols.items():
            expiry_dcf = self.get_dcf(expiry)
            slice_curves.append((expiry_dcf, SplineCurve(sorted(moneyness_vols))))
        return VolSurfaceSlices(self.date, slice_curves)
    
    def build_QD(self):
        implied_vols = self.build_implied()
        slice_params = []
        for (expiry, _), moneyness_vols in implied_vols.items():
            expiry_dcf = self.get_dcf(expiry)
            bounds = [(-1, 1), (0, 1)]
            moneyness_atm, vol_atm = moneyness_vols[0]
            delta_atm = fx_delta.get_delta_for_moneyness(moneyness=moneyness_atm, dcf=expiry_dcf,
                                                          sigma=vol_atm, delta_type=self._delta_type)
            res = optimize.minimize(self.get_QD_solver, x0=[0, 0],
                                    args=(moneyness_vols, delta_atm, vol_atm, expiry_dcf),
                                    method='L-BFGS-B', bounds=bounds)
            curve = PolynomialCurve([vol_atm] + list(res.x), delta_atm, self._delta_type)
            slice_params.append((expiry_dcf, curve))
        return VolSurfaceSlices(self.date, slice_params)
    
    def get_QD_solver(self, params: tuple[float], moneyness_vols: dict[float, float],
                        delta_atm: float, vol_atm: float, dcf: float) -> float:
        errors = []
        c1, c2 = params
        for moneyness, quote_vol in moneyness_vols:
            delta = fx_delta.get_delta_for_moneyness(moneyness=moneyness, dcf=dcf,
                                                      sigma=quote_vol, delta_type=self._delta_type)
            calc_vol = vol_atm + c1 * (delta - delta_atm) + c2 * (delta - delta_atm)**2
            errors.append(calc_vol-quote_vol)
        return np.sqrt(np.mean(np.array(errors)**2))
    
    def get_SABR_init(self, beta: float) -> list[float]:
        volvol, rho = 0.3, 0
        (expiry, fwd_price), quotes = next(iter(self._vol_quotes.items()))
        atm_vol = quotes[0]._value
        alpha = sabr.get_alpha(vol_atmf=atm_vol, forward_price=fwd_price, dcf=self.get_dcf(expiry),
                               volvol=volvol, beta=beta, rho=rho)
        return [volvol, rho, alpha]
    
    def get_SABR_solver(self, params: tuple[float], beta: float, market_vols: dict) -> float:
        volvol, rho, alpha = params
        errors = []
        for (expiry, fwd_price), moneyness_vols in market_vols.items():
            fwd_dcf = self.get_dcf(expiry)
            for strike, quote_vol in moneyness_vols:
                sabr_vol = sabr.get_vol(forward_price=fwd_price, strike=strike, dcf=fwd_dcf,
                                        alpha=alpha, volvol=volvol, beta=beta, rho=rho)
                errors.append(sabr_vol-quote_vol)
        return np.sqrt(np.mean(np.array(errors)**2))
    
    def build(self, surface_type: VolSurfaceType = None, **kwargs):
        match surface_type:
            case VolSurfaceType.SplineM | None:
                return self.build_SM()
            case VolSurfaceType.QuadD:
                return self.build_QD()
            case VolSurfaceType.SABR:
                market_vols = self.build_implied(OptionMoneynessType.Strike)
                return self.build_SABR(kwargs.get('beta', 0), market_vols)
            case _:
                raise Exception(f'{surface_type} not supported for FX')
    
    def get_graph_info(self, vol_surface: VolSurfaceSlices) -> tuple[list, list]:
        surface_points = []
        implied_points = []
        m_type = OptionMoneynessType.LogSimple
        for (expiry, f_price), moneyness_vols in self.build_implied(m_type).items():
            fwd_dcf = self.get_dcf(expiry)
            for moneyness, vol_quote in moneyness_vols:
                delta = fx_delta.get_delta_for_moneyness(moneyness, dcf=fwd_dcf,
                            sigma=vol_quote, delta_type=self._delta_type)
                strike = black_scholes.get_strike_for_moneyness(moneyness,
                            forward_price=f_price, moneyness_type=m_type)
                vol_calc = vol_surface.get_strike_vol(fwd_dcf, strike=strike, forward_price=f_price)
                surface_points.append((expiry, delta, vol_calc))
                implied_points.append((expiry, delta, vol_quote))
        return surface_points, implied_points
