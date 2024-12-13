from enum import StrEnum

class OptionType(StrEnum):
    Call = 'C'
    Put = 'P'

class ExerciseStyle(StrEnum):
    European = 'E'
    American = 'A'
    Bermudan = 'B'

class OptionGreekType(StrEnum):
    Delta_Black = 'D_BS'
    Gamma_Black = 'G_BS'
    Theta_Black = 'T_BS'
    Vega_Black = 'Ve_BS'
    Vanna_Black = 'Va_BS'
    Volga_Black = 'Vo_BS'
    Density = 'PDF'
    Delta_Adapted = 'D_Ad'
    Gamma_Adapted = 'G_Ad'
    # Vega_Adapted = 'Ve_Ad'
