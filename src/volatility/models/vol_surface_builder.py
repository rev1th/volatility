
from pydantic.dataclasses import dataclass
import datetime as dtm
import logging
from scipy import optimize
import numpy as np

from common.base_class import NameDateClass
from .option import CallOption, PutOption
from .vol_surface import VolSurfaceBase, VolSurfaceInterp, LV, SABR
from volatility.lib import option_analytics

logger = logging.Logger(__name__)


@dataclass
class VolSurfaceModel(NameDateClass):
    _option_chain: dict[tuple[dtm.date, float], tuple[CallOption, PutOption]]
    _rate: float = 0
    
    @property
    def rate(self) -> float:
        return self._rate
    
    def build_implied(self):
        res = []
        for (expiry, strike), (call_opt, put_opt) in self._option_chain.items():
            call_vol, put_vol = None, None
            if call_opt.is_valid():
                call_vol = call_opt.get_implied_vol(rate=self._rate)
                res.append([expiry, strike, call_vol])
            if put_opt.is_valid():
                put_vol = put_opt.get_implied_vol(rate=self._rate)
                res.append([expiry, strike, put_vol])
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
    
    def build_SABR(self, beta: float = 1):
        volvol, rho = 0.3, 0
        vs_implied = self.build_implied()
        underlier = next(iter(self._option_chain.values()))[0].underlying
        vol_atm = vs_implied.get_vol(underlier.expiry, underlier.price)
        alpha = option_analytics.get_SABR_alpha(vol_atmf=vol_atm, forward_price=underlier.price,
                                                dcf=underlier.get_expiry_dcf(),
                                                volvol=volvol, beta=beta, rho=rho)
        
        init_guess = [volvol, rho, alpha]
        bounds = [(1e-4, None), (-0.9999, 0.9999), (1e-4, None)]
        res = optimize.minimize(self.build_SABR_solver, x0=init_guess, args=(beta, vs_implied),
                                method='L-BFGS-B', bounds=bounds)
        volvol, rho, alpha = res.x
        return SABR(self.date, volvol=volvol, beta=beta, rho=rho, alpha=alpha)
    
    def build_SABR_solver(self, params: tuple[float], beta: float, vs_implied: VolSurfaceInterp) -> float:
        volvol, rho, alpha = params
        errors = []
        price_map = {}
        for expiry, strike, vol_implied in vs_implied.nodes:
            if expiry not in price_map:
                call_opt, _ = self._option_chain[expiry, strike]
                price_map[expiry] = call_opt.underlying.price
            sabr_vol = option_analytics.get_SABR_vol(
                        forward_price=price_map[expiry], strike=strike,
                        dcf=vs_implied.get_dcf(expiry),
                        alpha=alpha, volvol=volvol, beta=beta, rho=rho)
            errors.append(sabr_vol-vol_implied)
        return np.sqrt(np.mean(np.array(errors)**2))
    
    def build(self) -> VolSurfaceBase:
        return self.build_implied()
    
    def get_calibration_errors(self, vol_surface: VolSurfaceBase):
        res = []
        for (expiry, strike), (call_opt, put_opt) in self._option_chain.items():
            if call_opt.is_valid():
                call_error = call_opt.get_price(vol_surface, rate=self._rate) - call_opt.price
                res.append([expiry, strike, call_error])
            if put_opt.is_valid():
                put_error = put_opt.get_price(vol_surface, rate=self._rate) - put_opt.price
                res.append([expiry, strike, put_error])
        return res
    
    def get_underliers(self, vol_surface: VolSurfaceBase = None) -> dict[dtm.date, any]:
        res = dict()
        # for (expiry, _), (call_opt, _) in self._option_chain.items():
        for expiry, strike, _ in vol_surface.nodes:
            if expiry not in res:
                call_opt, _ = self._option_chain[expiry, strike]
                res[expiry] = call_opt.underlying
        return res
