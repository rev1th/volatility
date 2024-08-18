import datetime as dtm
import pandas as pd
from common import plotter

def get_vol_surface_figure(calc_points: list[tuple[dtm.date, float, float]],
                           node_points: list[tuple[dtm.date, float, float]]):
    col_names = ['Tenor', 'Strike', 'Volatility']
    vol_surface_df = pd.DataFrame(calc_points, columns=col_names)
    extra_df = pd.DataFrame(node_points, columns=col_names)
    return plotter.get_figure_3d(vol_surface_df, extra_df, title='Volatility Surface',
                                 data_name='Model Fitted', data2_name='Market Implied')

def display_vol_surface(calc_points: list[tuple[dtm.date, float, float]],
                        node_points: list[tuple[dtm.date, float, float]]) -> None:
    get_vol_surface_figure(calc_points, node_points).show()
