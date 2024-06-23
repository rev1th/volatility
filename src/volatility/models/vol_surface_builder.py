
from pydantic.dataclasses import dataclass
from typing import ClassVar
import datetime as dtm
import logging
from scipy import optimize
import numpy as np

from common.base_class import NameDateClass
from .option import CallOption, PutOption
from .vol_surface import VolSurfaceBase, VolSurfaceInterp, LV, SABR
from volatility.lib import option_analytics

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
    
    def build_SABR(self, beta: float = 1):
        init_guess = self.get_SABR_init(beta)
        bounds = [VOLVOL_BOUNDS, ALPHA_BOUNDS, RHO_BOUNDS]
        res = optimize.minimize(self.get_SABR_solver, x0=init_guess, args=(beta),
                                method='L-BFGS-B', bounds=bounds)
        volvol, alpha, rho = res.x
        return SABR(self.date, volvol=volvol, alpha=alpha, beta=beta, rho=rho)
    
    def build(self) -> VolSurfaceBase:
        return self.build_SABR()

@dataclass
class VolSurfaceModelListed(VolSurfaceModel):
    _option_chain: dict[tuple[dtm.date, float], tuple[CallOption, PutOption]]
    _rate: float = 0
    
    def build_implied(self):
        res = []
        for (expiry, _), (call_option, put_option) in self._option_chain.items():
            call_vol, put_vol = None, None
            if call_option.is_valid():
                call_vol = call_option.get_implied_vol(rate=self._rate)
                res.append([expiry, call_option.get_moneyness(), call_vol])
            if put_option.is_valid():
                put_vol = put_option.get_implied_vol(rate=self._rate)
                res.append([expiry, put_option.get_moneyness(), put_vol])
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
        return VolSurfaceInterp(self.date, res)
    
    def build_LV(self):
        vs_implied = self.build_implied()
        return LV(self.date, vs_implied._nodes, _rate=self._rate)
    
    def get_nearest_atm_option(self):
        atm_option = next(iter(self._option_chain.values()))[0]
        atm_price = atm_option.get_underlying_price()
        for (expiry, strike), cp_options in self._option_chain.items():
            if expiry != atm_option.expiry:
                break
            if abs(strike - atm_price) < abs(atm_option.strike - atm_price):
                atm_option = cp_options[0]
        return atm_option
    
    def get_SABR_init(self, beta: float):
        volvol, rho = 0.3, 0
        atm_option = self.get_nearest_atm_option()
        underlier_price = atm_option.get_underlying_price()
        expiry_dcf = atm_option.get_expiry_dcf()
        atm_vol = atm_option.get_implied_vol()
        alpha = option_analytics.get_SABR_alpha(
                    vol_atmf=atm_vol, forward_price=underlier_price, dcf=expiry_dcf,
                    volvol=volvol, beta=beta, rho=rho)
        return [volvol, alpha, rho]
    
    def get_SABR_solver(self, params: tuple[float], beta: float) -> float:
        volvol, alpha, rho = params
        errors = []
        expiry_map = {}
        for (expiry, strike), (call_option, put_option) in self._option_chain.items():
            if expiry not in expiry_map:
                expiry_map[expiry] = (call_option.get_underlying_price(), call_option.underlying.get_expiry_dcf())
            sabr_vol = option_analytics.get_SABR_vol(
                        forward_price=expiry_map[expiry][0], strike=strike, dcf=expiry_map[expiry][1],
                        alpha=alpha, volvol=volvol, beta=beta, rho=rho)
            if call_option.is_valid():
                price_err = call_option.get_price(sabr_vol) - call_option.price
                errors.append(price_err)
            if put_option.is_valid():
                price_err = put_option.get_price(sabr_vol) - put_option.price
                errors.append(price_err)
        return np.sqrt(np.mean(np.array(errors)**2))
    
    def build(self):
        return self.build_implied()
    
    def get_calibration_errors(self, vol_surface: VolSurfaceBase):
        res = []
        for (expiry, strike), (call_opt, put_opt) in self._option_chain.items():
            if call_opt.is_valid():
                call_error = call_opt.get_price(vol_surface, rate=self._rate) - call_opt.price
                res.append([expiry, strike, 'C', call_error])
            if put_opt.is_valid():
                put_error = put_opt.get_price(vol_surface, rate=self._rate) - put_opt.price
                res.append([expiry, strike, 'P', put_error])
        return res
    
    def get_graph_info(self, vol_surface: VolSurfaceBase) -> list[tuple[dtm.date, float, float]]:
        expiry_map = {}
        surface_points = []
        implied_points = []
        for (expiry, strike), (call_option, put_option) in self._option_chain.items():
            if expiry not in expiry_map:
                price = call_option.get_underlying_price()
                dcf = call_option.underlying.get_expiry_dcf()
                surface_points.append((expiry, price, vol_surface.get_strike_vol(dcf, price, price)))
                expiry_map[expiry] = (dcf, price)
            vol = vol_surface.get_strike_vol(expiry_map[expiry][0], strike, expiry_map[expiry][1])
            surface_points.append((expiry, strike, vol))
            if call_option.is_valid():
                implied_points.append((expiry, strike, call_option.get_implied_vol()))
            if put_option.is_valid():
                implied_points.append((expiry, strike, put_option.get_implied_vol()))
        return surface_points, implied_points
