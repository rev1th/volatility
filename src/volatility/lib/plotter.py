import datetime as dtm
import pandas as pd
from common.app import plotter

def get_surface_figure(data_points: list[tuple[dtm.date, float, float, float]],
                           col_names: list[str], title: str = 'Volatility Surface', **kwargs):
    vol_surface_df = pd.DataFrame(data_points, columns=col_names)
    vol_surface_df.sort_values(by=col_names[:2], inplace=True)
    return plotter.get_figure_3d(vol_surface_df, title=title, **kwargs)

def display_surface(data_points: list[tuple[dtm.date, float, float]],
                        col_names: list[str], **kwargs) -> None:
    get_surface_figure(data_points, col_names, **kwargs).show()
