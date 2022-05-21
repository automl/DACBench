import json
import os
from setuptools import setup, find_packages


def get_other_requirements():
    other_requirements = {}
    for file in os.listdir('./other_requirements'):
        with open(f'./other_requirements/{file}', encoding='utf-8') as rq:
            requirements = json.load(rq)
            other_requirements.update(requirements)
            return other_requirements


setup(
    packages=find_packages(exclude=['tests', 'examples', 'dacbench.wrappers.*', 'dacbench.envs.fast-downward/*']),
)
