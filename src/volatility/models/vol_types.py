from enum import StrEnum

class VolatilityModelType(StrEnum):
    PolyMoneyness = 'StickyMoneyness'
    LV = 'LocalVol'
    SABR = 'SABR'
    PolyDelta = 'StickyDelta'

class VolatilityQuoteType(StrEnum):
    ATM = 'ATM'
    Butterfly = 'BF'
    RiskReversal = 'RR'
    Call = 'CALL'
    Put = 'PUT'
