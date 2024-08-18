from enum import StrEnum

class VolatilityModelType(StrEnum):
    Interp = 'Interpolated'
    LV = 'LocalVol'
    SABR = 'SABR'
    QuadD = 'QuadraticDelta'
    SplineM = 'SplineMoneyness'

class VolatilityQuoteType(StrEnum):
    ATM = 'ATM'
    Butterfly = 'BF'
    RiskReversal = 'RR'
    Call = 'CALL'
    Put = 'PUT'
