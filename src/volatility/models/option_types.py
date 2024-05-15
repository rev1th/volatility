
from pydantic.dataclasses import dataclass
from enum import StrEnum


@dataclass
class OptionType(StrEnum):
    Call = 'C'
    Put = 'P'

@dataclass
class OptionStyle(StrEnum):
    European = 'E'
    American = 'A'
    Bermudan = 'B'

@dataclass
class OptionMoneyness(StrEnum):
    ATM = 'ATM'
    _10Delta = '10D'
    _25Delta = '25D'
