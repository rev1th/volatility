from enum import StrEnum

class OptionMoneynessType(StrEnum):
    Strike = 'Strike'
    Simple = 'Simple'
    LogSimple = 'LogSimple'
    Normal = 'Normal'
    Standard = 'Standard'

class ATMStrikeType(StrEnum):
    Forward = 'Forward'
    DeltaNeutral = 'DeltaNeutral'

class NumeraireConvention(StrEnum):
    Regular = 'Regular'
    Inverse = 'Inverse'
