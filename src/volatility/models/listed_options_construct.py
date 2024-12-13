from pydantic.dataclasses import dataclass
import datetime as dtm
import numpy as np
import logging

from common.numeric import solver
from volatility.instruments.option import CallOption, PutOption
from volatility.instruments.option_types import OptionGreekType
from volatility.lib import black_scholes, sabr
from .delta_types import OptionMoneynessType
from .vol_types import VolatilityModelType
from .vol_surface import VolSurfaceBase, VolSurfaceSlices, PolynomialMoneynessCurve
from .vol_surface_lv import LocalVol
from .vol_surface_sabr import SABRSlice
from .vol_surface_construct import VolSurfaceConstruct

logger = logging.Logger(__name__)

@dataclass
class ModelStrikeLine:
    strike: float
    call: CallOption
    put: PutOption
    c_weight: float = 1
    p_weight: float = 1

@dataclass
class ModelStrikeSlice:
    expiry: dtm.date
    discount_factor: float
    lines: list[ModelStrikeLine]

@dataclass
class ListedOptionsConstruct(VolSurfaceConstruct):
    _option_matrix: list[ModelStrikeSlice]
    
    def __post_init__(self):
        # normalize weights using only OTM options
        for strike_slice in self._option_matrix:
            wsum = 0
            forward_price = strike_slice.lines[0].call.get_forward_price(self.date)
            for strike_line in strike_slice.lines:
                if strike_line.call.is_valid_price(self.date) and strike_line.strike >= forward_price:
                    wsum += strike_line.c_weight
                else:
                    strike_line.c_weight = 0
                if strike_line.put.is_valid_price(self.date) and strike_line.strike <= forward_price:
                    wsum += strike_line.p_weight
                else:
                    strike_line.p_weight = 0
            if wsum > 0:
                for strike_line in strike_slice.lines:
                    strike_line.c_weight /= wsum
                    strike_line.p_weight /= wsum
    
    def get_implied(self, moneyness_type: OptionMoneynessType) -> dict[float, list[tuple[float, float]]]:
        slices = {}
        date = self.date
        for strike_slice in self._option_matrix:
            df = strike_slice.discount_factor
            ref_option = strike_slice.lines[0].call
            dcf, forward_price = ref_option.get_expiry_dcf(date), ref_option.get_forward_price(date)
            nodes = []
            for strike_line in strike_slice.lines:
                moneyness = black_scholes.get_moneyness(forward_price=forward_price,
                    strike=strike_line.strike, tau=dcf, moneyness_type=moneyness_type)
                if strike_line.c_weight > 0:
                    call_vol = strike_line.call.get_implied_volatility(date, discount_factor=df)
                    nodes.append((moneyness, call_vol, strike_line.c_weight))
                if strike_line.p_weight > 0:
                    put_vol = strike_line.put.get_implied_volatility(date, discount_factor=df)
                    nodes.append((moneyness, put_vol, strike_line.p_weight))
            slices[dcf] = nodes
        return slices
    
    def build_LV(self):
        m_type = OptionMoneynessType.Normal
        slice_nodes = self.get_implied(m_type)
        return LocalVol(self.date, slice_nodes, m_type, name=self.name)
    
    def get_nearest_atm_option(self) -> CallOption:
        atm_option = None
        for strike_slice in self._option_matrix:
            atm_price = strike_slice.lines[0].call.get_forward_price(self.date)
            for strike_line in strike_slice.lines:
                if not atm_option or abs(strike_line.strike - atm_price) < abs(atm_option.strike - atm_price):
                    if strike_line.c_weight > 0:
                        atm_option = strike_line.call
                    elif strike_line.p_weight > 0:
                        atm_option = strike_line.put
            if atm_option:
                return atm_option
        raise Exception(f'No valid option strikes near {atm_price}')
    
    def get_SABR_init(self, beta: float):
        volvol, rho = 0.3, 0
        atm_option = self.get_nearest_atm_option()
        forward_price = atm_option.get_forward_price(self.date)
        expiry_dcf = atm_option.get_expiry_dcf(self.date)
        atm_vol = atm_option.get_implied_volatility(self.date)
        alpha = sabr.get_alpha(
                    vol_atmf=atm_vol, forward_price=forward_price, tau=expiry_dcf,
                    volvol=volvol, beta=beta, rho=rho)
        return [alpha, volvol, rho]
    
    def get_SABR_solver(self, params: tuple[float], beta: float, strike_slice: ModelStrikeSlice) -> float:
        alpha, volvol, rho = params
        errors = []
        ref_option = strike_slice.lines[0].call
        fwd_price = ref_option.get_forward_price(self.date)
        expiry_dcf = ref_option.get_expiry_dcf(self.date)
        for strike_line in strike_slice.lines:
            sabr_vol = sabr.get_vol(
                        forward_price=fwd_price, strike=strike_line.strike, tau=expiry_dcf,
                        alpha=alpha, volvol=volvol, beta=beta, rho=rho)
            if strike_line.c_weight > 0:
                price_calc = strike_line.call.get_price_for_vol(self.date, sabr_vol,
                                discount_factor=strike_slice.discount_factor)
                price_err = price_calc - strike_line.call.data[self.date]
                errors.append(strike_line.c_weight * price_err)
            if strike_line.p_weight > 0:
                price_calc = strike_line.put.get_price_for_vol(self.date, sabr_vol,
                                discount_factor=strike_slice.discount_factor)
                price_err = price_calc - strike_line.put.data[self.date]
                errors.append(strike_line.p_weight * price_err)
        return np.sqrt(np.sum(np.array(errors)**2))
    
    def build_SABR(self, beta: float, *args):
        init_guess = self.get_SABR_init(beta)
        bounds = [sabr.ALPHA_BOUNDS, sabr.VOLVOL_BOUNDS, sabr.RHO_BOUNDS]
        slices = []
        for strike_slice in self._option_matrix:
            expiry_dcf = strike_slice.lines[0].call.get_expiry_dcf(self.date)
            res = solver.find_fit(self.get_SABR_solver, init_guess=init_guess, bounds=bounds,
                                  args=(beta, strike_slice, *args))
            alpha, volvol, rho = res
            slices.append(SABRSlice(expiry_dcf, alpha=alpha, beta=beta, rho=rho, volvol=volvol))
        return VolSurfaceSlices(self.date, slices, name=self.name)
    
    def build_PM(self):
        m_type = OptionMoneynessType.LogSimple
        slice_curves = []
        for dcf, nodes in self.get_implied(m_type).items():
            if len(nodes) > 3:
                try:
                    slice_curves.append(PolynomialMoneynessCurve(dcf, nodes, m_type))
                except Exception as ex:
                    logger.error(f'Slice {dcf} for {self.name} failed: {ex}')
        if slice_curves:
            return VolSurfaceSlices(self.date, slice_curves, name=self.name)
        logger.critical(f'No valid slices to build Surface {self.name}')
    
    def build(self, model_type: VolatilityModelType = None, **kwargs) -> VolSurfaceBase:
        match model_type:
            case VolatilityModelType.PolyMoneyness | None:
                return self.build_PM()
            case VolatilityModelType.LV:
                return self.build_LV()
            case VolatilityModelType.SABR:
                return self.build_SABR(beta=kwargs.get('beta', 1))
            case _:
                raise Exception(f'{model_type} not supported for listed options')
    
    def get_calibration_summary(self, vol_surface: VolSurfaceBase):
        res = []
        for strike_slice in self._option_matrix:
            expiry, df = strike_slice.expiry, strike_slice.discount_factor
            for strike_line in strike_slice.lines:
                if strike_line.call.is_valid_price(self.date):
                    price = strike_line.call.data[self.date]
                    price_calc = strike_line.call.get_price(vol_surface, discount_factor=df)
                    res.append((expiry, strike_line.strike, 'Call', price, price_calc-price))
                if strike_line.put.is_valid_price(self.date):
                    price = strike_line.put.data[self.date]
                    price_calc = strike_line.put.get_price(vol_surface, discount_factor=df)
                    res.append((expiry, strike_line.strike, 'Put', price, price_calc-price))
        return res, ['Expiry', 'Strike', 'Type', 'Price', 'Error']
    
    def get_vols_graph(self, vol_surface: VolSurfaceBase) -> tuple[list[tuple[dtm.date, float]], list[str]]:
        surface_points = []
        for strike_slice in self._option_matrix:
            expiry, df = strike_slice.expiry, strike_slice.discount_factor
            ref_option = strike_slice.lines[0].call
            fwd_price = ref_option.get_forward_price(self.date)
            expiry_dcf = ref_option.get_expiry_dcf(self.date)
            vol_atm = vol_surface.get_strike_vol(expiry_dcf, fwd_price, fwd_price)
            surface_points.append((expiry, fwd_price, vol_atm, None, None))
            for strike_line in strike_slice.lines:
                vol_k = vol_surface.get_strike_vol(expiry_dcf, strike_line.strike, fwd_price)
                row = [expiry, strike_line.strike, vol_k, None, None]
                if strike_line.call.is_valid_price(self.date):
                    row[3] = strike_line.call.get_implied_volatility(self.date, discount_factor=df)
                if strike_line.put.is_valid_price(self.date):
                    row[4] = strike_line.put.get_implied_volatility(self.date, discount_factor=df)
                surface_points.append(tuple(row))
        return surface_points, ['Tenor', 'Strike', 'Model Fitted', 'Market Implied Call', 'Market Implied Put']
    
    def get_greeks_graph(self, vol_surface: VolSurfaceBase) -> tuple[list[tuple[dtm.date, float]], list[str]]:
        surface_greeks = []
        greek_types = [gt for gt in OptionGreekType]
        for strike_slice in self._option_matrix:
            expiry, df = strike_slice.expiry, strike_slice.discount_factor
            for strike_line in strike_slice.lines:
                call_greeks = strike_line.call.get_greeks(vol_surface, greek_types, discount_factor=df)
                surface_greeks.append((expiry, strike_line.strike, *call_greeks))
                put_greeks = strike_line.put.get_greeks(vol_surface, greek_types, discount_factor=df)
                surface_greeks.append((expiry, strike_line.strike, *put_greeks))
        return surface_greeks, ['Tenor', 'Strike'] + [gt.name for gt in greek_types]
