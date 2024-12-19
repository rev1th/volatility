from pydantic.dataclasses import dataclass
import datetime as dtm
import numpy as np

from common.numeric import solver
from common.models.base_instrument import BaseInstrument
from .construct_types import NumeraireConvention, ATMStrikeType, OptionMoneynessType
from .vol_surface_construct import VolSurfaceConstruct
from .vol_surface import VolSurfaceSlices, VolStrikeSlice, PolynomialMoneynessCurve, ROOT_EPS
from .vol_surface_sabr import SABRSurface
from .vol_types import VolatilityQuoteType, VolatilityModelType
from volatility.instruments.fx_option import FXCallOption, FXPutOption
from volatility.instruments.option_types import OptionGreekType
from volatility.lib import fx_convention, sabr


@dataclass
class FXVolQuote:
    _type: VolatilityQuoteType
    value: float
    delta: float | None = None
    weight: float = 1

@dataclass
class FXDeltaSlice:
    expiry: dtm.date
    forward_price: float
    quotes: list[FXVolQuote]
    discount_factor: float = 1

# implied objects for internal use
@dataclass
class FXQuoteImplied:
    moneyness: float
    value: float
    weight: float = 1

@dataclass
class FXSliceImplied:
    forward_price: float
    tau: float
    quotes: list[FXQuoteImplied]
    expiry: dtm.date

@dataclass
class FXDeltaCurve(VolStrikeSlice):
    _quotes: list[FXQuoteImplied]
    _numeraire: NumeraireConvention

    def __post_init__(self):
        xs, ys, ws = zip(*[(q.moneyness, q.value, q.weight) for q in self._quotes])
        # Quadratic function
        self._polynomial = np.polynomial.Polynomial.fit(xs, ys, w=ws, deg=2)
    
    def get_strike_vol(self, tau: float, strike: float, forward_price: float):
        delta, d_iter = 0.5, None
        numeraire, polynomial = self._numeraire, self._polynomial
        while(not d_iter or abs(delta - d_iter) > ROOT_EPS):
            sigma = polynomial(delta)
            d_iter = delta
            delta = fx_convention.get_delta(forward_price=forward_price,
                strike=strike, tau=tau, sigma=sigma, numeraire=numeraire)
        return sigma


@dataclass
class FXVolSurfaceConstruct(VolSurfaceConstruct):
    _slices: list[FXDeltaSlice]
    _numeraire: NumeraireConvention
    _atm_type = ATMStrikeType.DeltaNeutral

    def get_dcf(self, date: dtm.date) -> float:
        return (date - self.date).days / 365
    
    def get_implied(self, moneyness_type: OptionMoneynessType = None) -> list[FXSliceImplied]:
        slices_implied = []
        numeraire = self._numeraire
        for slice in self._slices:
            expiry, fwd_price = slice.expiry, slice.forward_price
            dcf = self.get_dcf(expiry)
            bf_vols, bf_ws = {}, {}
            rr_vols, rr_ws = {}, {}
            moneyness_vols = []
            for quote in slice.quotes:
                match quote._type:
                    case VolatilityQuoteType.ATM:
                        atm_vol, atm_weight = quote.value, quote.weight
                    case VolatilityQuoteType.RiskReversal:
                        rr_vols[quote.delta] = quote.value
                        rr_ws[quote.delta] = quote.weight
                    case VolatilityQuoteType.Butterfly:
                        bf_vols[quote.delta] = quote.value
                        bf_ws[quote.delta] = quote.weight
            if moneyness_type:
                x_atm = fx_convention.get_moneyness_atm(atm_type=self._atm_type, tau=dcf, sigma=atm_vol,
                            forward_price=fwd_price, numeraire=numeraire, moneyness_type=moneyness_type)
            else:
                x_atm = fx_convention.get_delta_atm(atm_type=self._atm_type, tau=dcf, sigma=atm_vol,
                            numeraire=numeraire)
            moneyness_vols.append(FXQuoteImplied(x_atm, atm_vol, atm_weight))
            for delta in bf_vols:
                d_weight = 1 / (1 / atm_weight + 1 / bf_ws[delta] + 1 / (2*rr_ws[delta]))
                put_vol = atm_vol + bf_vols[delta] - rr_vols[delta] / 2
                put_x = -delta
                if moneyness_type:
                    put_x = fx_convention.get_moneyness_for_delta(put_x, tau=dcf, sigma=put_vol,
                                forward_price=fwd_price, numeraire=numeraire, moneyness_type=moneyness_type)
                else:
                    put_x = fx_convention.get_delta_complement(put_x, tau=dcf, sigma=put_vol,
                                numeraire=numeraire)
                moneyness_vols.append(FXQuoteImplied(put_x, put_vol, d_weight))
                call_vol = atm_vol + bf_vols[delta] + rr_vols[delta] / 2
                call_x = delta
                if moneyness_type:
                    call_x = fx_convention.get_moneyness_for_delta(call_x, tau=dcf, sigma=call_vol,
                                forward_price=fwd_price, numeraire=numeraire, moneyness_type=moneyness_type)
                moneyness_vols.append(FXQuoteImplied(call_x, call_vol, d_weight))
            slices_implied.append(FXSliceImplied(fwd_price, dcf, moneyness_vols, expiry))
        return slices_implied
    
    def build_PM(self):
        m_type = OptionMoneynessType.Standard
        implied_slices = self.get_implied(m_type)
        slice_curves = []
        for slice in implied_slices:
            moneyness_vols = [(fvi.moneyness, fvi.value, fvi.weight) for fvi in slice.quotes]
            slice_curves.append(PolynomialMoneynessCurve(slice.tau, sorted(moneyness_vols), _moneyness_type=m_type))
        return VolSurfaceSlices(self.date, slice_curves)
    
    def build_PD(self):
        implied_slices = self.get_implied()
        slice_params = []
        for slice in implied_slices:
            dcf, quotes = slice.tau, slice.quotes
            delta_curve = FXDeltaCurve(dcf, quotes, self._numeraire)
            slice_params.append(delta_curve)
        return VolSurfaceSlices(self.date, slice_params)
    
    def get_SABR_init(self, beta: float) -> list[float]:
        volvol, rho = 0.3, 0
        ref_slice = self._slices[0]
        dcf = self.get_dcf(ref_slice.expiry)
        price, vol_atm = ref_slice.forward_price, ref_slice.quotes[0].value
        alpha = sabr.get_alpha(vol_atmf=vol_atm, forward_price=price, tau=dcf, volvol=volvol, beta=beta, rho=rho)
        return [alpha, volvol, rho]
    
    def get_SABR_solver(self, params: tuple[float], beta: float, implied_slices: list[FXSliceImplied]) -> float:
        alpha, volvol, rho = params
        errors = []
        error_weights = []
        for slice in implied_slices:
            price, dcf = slice.forward_price, slice.tau
            for quote in slice.quotes:
                sabr_vol = sabr.get_vol(forward_price=price, strike=quote.moneyness, tau=dcf,
                                        alpha=alpha, volvol=volvol, beta=beta, rho=rho)
                errors.append(sabr_vol-quote.value)
                error_weights.append(quote.weight)
        return np.sqrt(np.dot(error_weights, np.array(errors)**2) / np.sum(error_weights))
    
    def build_SABR(self, beta: float = 1):
        init_guess = self.get_SABR_init(beta)
        market_vols = self.get_implied(OptionMoneynessType.Strike)
        bounds = [sabr.ALPHA_BOUNDS, sabr.VOLVOL_BOUNDS, sabr.RHO_BOUNDS]
        res = solver.find_fit(self.get_SABR_solver, init_guess=init_guess, bounds=bounds,
                              args=(beta, market_vols))
        alpha, volvol, rho = res
        return SABRSurface(self.date, volvol=volvol, alpha=alpha, beta=beta, rho=rho)
    
    def build(self, model_type: VolatilityModelType = None, **kwargs):
        match model_type:
            case VolatilityModelType.PolyMoneyness | None:
                return self.build_PM()
            case VolatilityModelType.PolyDelta:
                return self.build_PD()
            case VolatilityModelType.SABR:
                return self.build_SABR(**kwargs)
            case _:
                raise Exception(f'{model_type} not supported for FX quotes')
    
    def get_vols_graph(self, vol_surface: VolSurfaceSlices) -> tuple[list, list[str]]:
        surface_points = []
        for slice in self.get_implied():
            expiry, price, dcf = slice.expiry, slice.forward_price, slice.tau
            for quote in slice.quotes:
                delta = quote.moneyness
                vol_quote = quote.value
                strike = fx_convention.get_strike_for_delta(delta, forward_price=price, tau=dcf,
                            sigma=vol_quote, numeraire=self._numeraire)
                vol_calc = vol_surface.get_strike_vol(dcf, strike=strike, forward_price=price)
                surface_points.append((expiry, delta, vol_calc, vol_quote))
        return surface_points, ['Expiry', 'Delta', 'Model Fitted', 'Market Quotes']
    
    def get_calibration_summary(self, vol_surface: VolSurfaceSlices):
        res = []
        for slice in self.get_implied():
            expiry, price, dcf = slice.expiry, slice.forward_price, slice.tau
            for quote in slice.quotes:
                delta = quote.moneyness
                vol_quote = quote.value
                strike = fx_convention.get_strike_for_delta(delta, forward_price=price, tau=dcf,
                            sigma=vol_quote, numeraire=self._numeraire)
                vol_calc = vol_surface.get_strike_vol(dcf, strike=strike, forward_price=price)
                res.append((expiry, delta, vol_quote, vol_calc-vol_quote))
        return res, ['Expiry', 'Delta', 'Quote', 'Error']
    
    def get_greeks_graph(self, vol_surface: VolSurfaceSlices) -> tuple[list[tuple[dtm.date, float]], list[str]]:
        surface_greeks = []
        greek_types = [gt for gt in OptionGreekType]
        for slice in self.get_implied():
            expiry, price, dcf = slice.expiry, slice.forward_price, slice.tau
            underlier = BaseInstrument()
            underlier.data[self.date] = price
            for quote in slice.quotes:
                delta = quote.moneyness
                vol_quote = quote.value
                strike = fx_convention.get_strike_for_delta(delta, forward_price=price, tau=dcf,
                            sigma=vol_quote, numeraire=self._numeraire)
                call = FXCallOption(underlier, expiry, strike, expiry)
                put = FXPutOption(underlier, expiry, strike, expiry)
                call_greeks = call.get_greeks(vol_surface, greek_types)
                surface_greeks.append((expiry, strike, *[call_greeks[gt] for gt in greek_types]))
                put_greeks = put.get_greeks(vol_surface, greek_types)
                surface_greeks.append((expiry, strike, *[put_greeks[gt] for gt in greek_types]))
        return surface_greeks, ['Tenor', 'Strike'] + [gt.name for gt in greek_types]
