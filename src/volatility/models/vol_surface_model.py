from pydantic.dataclasses import dataclass
import datetime as dtm
import logging
import numpy as np

from common.base_class import NameDateClass
from common.numeric import solver
from .delta_types import OptionMoneynessType
from .vol_types import VolatilityModelType
from .vol_surface import VolSurfaceBase, VolSurfaceInterpolation, LocalVol, SABR
from volatility.instruments.option import CallOption, PutOption
from volatility.lib import sabr

logger = logging.Logger(__name__)

RHO_BOUNDS = (-0.9999, 0.9999)
ALPHA_BOUNDS = VOLVOL_BOUNDS = (1e-4, None)

@dataclass
class VolSurfaceModel(NameDateClass):
    
    def build_implied(self):
        '''Build implied vol surface from market quotes'''
    
    def get_SABR_solver(self, _: tuple[float], __: float) -> float:
        '''Solver error for SABR calibration'''
    
    def get_SABR_init(self, _: float) -> list[float]:
        '''Get initial params for SABR calibration'''
    
    def build_SABR(self, beta: float, *args):
        init_guess = self.get_SABR_init(beta)
        bounds = [VOLVOL_BOUNDS, ALPHA_BOUNDS, RHO_BOUNDS]
        res = solver.find_fit(self.get_SABR_solver, init_guess=init_guess, args=(beta, *args), bounds=bounds)
        volvol, alpha, rho = res
        return SABR(self.date, volvol=volvol, alpha=alpha, beta=beta, rho=rho)
    
    def build(self, _: VolatilityModelType = None) -> VolSurfaceBase:
        return self.build_SABR()

@dataclass
class VolSurfaceModelListed(VolSurfaceModel):
    _option_chain: dict[tuple[dtm.date, float], tuple[CallOption, PutOption]]
    _rate: float = 0
    
    def build_implied(self):
        res = []
        m_type = OptionMoneynessType.Normal
        for (expiry, _), (call_option, put_option) in self._option_chain.items():
            call_vol, put_vol = None, None
            if call_option.is_valid_price(self.date):
                call_vol = call_option.get_implied_vol(self.date, rate=self._rate)
                res.append([expiry, call_option.get_moneyness(self.date, m_type), call_vol])
            if put_option.is_valid_price(self.date):
                put_vol = put_option.get_implied_vol(self.date, rate=self._rate)
                res.append([expiry, put_option.get_moneyness(self.date, m_type), put_vol])
            # strike_vol = None
            # if call_vol:
            #     if put_vol:
            #         strike_vol = (call_vol + put_vol)/2
            #     else:
            #         strike_vol = call_vol
            # elif put_vol:
            #     strike_vol = put_vol
            # if strike_vol:
            #     res.append([expiry, strike, strike_vol])
        return VolSurfaceInterpolation(self.date, res, m_type)
    
    def build_LV(self):
        vs_implied = self.build_implied()
        return LocalVol(self.date, vs_implied._nodes, vs_implied._moneyness_type, _rate=self._rate)
    
    def get_nearest_atm_option(self):
        atm_option = None
        atm_price = next(iter(self._option_chain.values()))[0].get_underlying_price(self.date)
        for (_, strike), cp_options in self._option_chain.items():
            if not atm_option or abs(strike - atm_price) < abs(atm_option.strike - atm_price):
                if cp_options[0].is_valid_price(self.date):
                    atm_option = cp_options[0]
                elif cp_options[1].is_valid_price(self.date):
                    atm_option = cp_options[1]
        if not atm_option:
            raise Exception(f'No valid option strikes near {atm_price}')
        return atm_option
    
    def get_SABR_init(self, beta: float):
        volvol, rho = 0.3, 0
        atm_option = self.get_nearest_atm_option()
        underlier_price = atm_option.get_underlying_price(self.date)
        expiry_dcf = atm_option.get_expiry_dcf(self.date)
        atm_vol = atm_option.get_implied_vol(self.date)
        alpha = sabr.get_alpha(
                    vol_atmf=atm_vol, forward_price=underlier_price, dcf=expiry_dcf,
                    volvol=volvol, beta=beta, rho=rho)
        return [volvol, alpha, rho]
    
    def get_SABR_solver(self, params: tuple[float], beta: float) -> float:
        volvol, alpha, rho = params
        errors = []
        expiry_map = {}
        for (expiry, strike), (call_option, put_option) in self._option_chain.items():
            if expiry not in expiry_map:
                fwd_price = call_option.get_underlying_price(self.date)
                if fwd_price:
                    expiry_map[expiry] = (fwd_price, call_option.get_expiry_dcf(self.date))
                else:
                    expiry_map[expiry] = None
            if not expiry_map[expiry]:
                continue
            sabr_vol = sabr.get_vol(
                        forward_price=expiry_map[expiry][0], strike=strike, dcf=expiry_map[expiry][1],
                        alpha=alpha, volvol=volvol, beta=beta, rho=rho)
            if call_option.is_valid_price(self.date):
                price_err = call_option.get_price_for_vol(self.date, sabr_vol) - call_option.data[self.date]
                errors.append(price_err)
            if put_option.is_valid_price(self.date):
                price_err = put_option.get_price_for_vol(self.date, sabr_vol) - put_option.data[self.date]
                errors.append(price_err)
        return np.sqrt(np.mean(np.array(errors)**2))
    
    def build(self, surface_type: VolatilityModelType = None, **kwargs) -> VolSurfaceBase:
        match surface_type:
            case VolatilityModelType.LV | None:
                return self.build_LV()
            case VolatilityModelType.SABR:
                return self.build_SABR(beta=kwargs.get('beta', 1))
            case VolatilityModelType.Interp:
                return self.build_implied()
            case _:
                raise Exception(f'{surface_type} not supported for listed strikes')
    
    def get_calibration_errors(self, vol_surface: VolSurfaceBase):
        res = []
        for (expiry, strike), (call_opt, put_opt) in self._option_chain.items():
            if call_opt.is_valid_price(self.date):
                call_error = call_opt.get_price(vol_surface, rate=self._rate) - call_opt.data[self.date]
                res.append([expiry, strike, 'C', call_error])
            if put_opt.is_valid_price(self.date):
                put_error = put_opt.get_price(vol_surface, rate=self._rate) - put_opt.data[self.date]
                res.append([expiry, strike, 'P', put_error])
        return res
    
    def get_graph_info(self, vol_surface: VolSurfaceBase) -> tuple[list, list]:
        expiry_map = {}
        surface_points = []
        implied_points = []
        for (expiry, strike), (call_option, put_option) in self._option_chain.items():
            if expiry not in expiry_map:
                fwd_price = call_option.get_underlying_price(self.date)
                if fwd_price:
                    dcf = call_option.get_expiry_dcf(self.date)
                    vol_atm = vol_surface.get_strike_vol(dcf, fwd_price, fwd_price)
                    surface_points.append((expiry, fwd_price, vol_atm))
                    expiry_map[expiry] = (dcf, fwd_price)
                else:
                    expiry_map[expiry] = None
            if not expiry_map[expiry]:
                continue
            vol_k = vol_surface.get_strike_vol(expiry_map[expiry][0], strike, expiry_map[expiry][1])
            surface_points.append((expiry, strike, vol_k))
            if call_option.is_valid_price(self.date):
                implied_points.append((expiry, strike, call_option.get_implied_vol(self.date)))
            if put_option.is_valid_price(self.date):
                implied_points.append((expiry, strike, put_option.get_implied_vol(self.date)))
        return surface_points, implied_points
