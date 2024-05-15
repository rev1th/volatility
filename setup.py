
from setuptools import setup, find_packages
import os

requirements_path = f"{os.path.dirname(__file__)}/requirements.txt"
install_requires = []
with open(requirements_path) as f:
    install_requires = f.read().splitlines()

setup(
    name='volatility',
    version='1.0',
    packages = find_packages('src'),  # Automatically find the packages that are recognized in the '__init__.py'.
    package_dir={"": "src"},
    install_requires=install_requires
)
