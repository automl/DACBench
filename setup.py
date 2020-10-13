from os.path import dirname, abspath, join
from setuptools import setup, find_packages
import toml

with open("pyproject.toml", "r") as f:
    requirements = toml.loads(f.read())

prod = requirements["install_requires"]
dev = requirements["dev-dependencies"]

setup(
    name="DAClib",
    install_requires=[x for x in prod],
    extras_require={"dev": [x for x in dev]},
)
