# Description

* Volatility surface modelling (LocalVol and SABR) for listed options
* FX Volatlity surface construction from market quotes
<br/><br/>

# Setup
```
python setup.py clean --all
python -m build --wheel
pip install --force-reinstall --no-deps git+https://github.com/rev1th/volatility@main#egg=volatility
```
