from enum import StrEnum

class OptionType(StrEnum):
    Call = 'C'
    Put = 'P'

class ExerciseStyle(StrEnum):
    European = 'E'
    American = 'A'
    Bermudan = 'B'
