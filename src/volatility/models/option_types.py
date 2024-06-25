from enum import StrEnum


class OptionType(StrEnum):
    Call = 'C'
    Put = 'P'

class OptionStyle(StrEnum):
    European = 'E'
    American = 'A'
    Bermudan = 'B'

class OptionMoneynessType(StrEnum):
    Strike = 'Strike'
    Simple = 'Simple'
    LogSimple = 'LogSimple'
    Normal = 'Normal'
    Standard = 'Standard'

class ATMStrikeType(StrEnum):
    Forward = 'Forward'
    DN = 'DeltaNeutral'

class FXDeltaType(StrEnum):
    Spot = 'Spot'
    Forward = 'Forward'
    SpotPremium = 'SpotPremium'
    ForwardPremium = 'ForwardPremium'
