
from enum import StrEnum


class OptionType(StrEnum):
    Call = 'C'
    Put = 'P'

class OptionStyle(StrEnum):
    European = 'E'
    American = 'A'
    Bermudan = 'B'

class VolatilityQuoteType(StrEnum):
    ATM = 'ATM'
    Butterfly = 'BF'
    RiskReversal = 'RR'
    Call = 'CALL'
    Put = 'PUT'
