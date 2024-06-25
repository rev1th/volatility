# Description

* Volatility surface modelling (LocalVol and SABR) for listed options
* FX Volatlity surface construction from market quotes
<br/><br/>

# Setup
```
python setup.py clean --all bdist_wheel
pip uninstall volatility -y
pip install ..\volatility\dist\volatility-1.0-py3-none-any.whl
```
