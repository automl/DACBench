import json
import os
from setuptools import setup, find_packages
from pyparsing import *


def get_other_requirements():
    other_requirements = {}
    for file in os.listdir('./dacbench/container/other_requirements'):
        with open(f'./dacbench/container/other_requirements/{file}', encoding='utf-8') as rq:
            requirements = json.load(rq)
            other_requirements.update(requirements)
            return other_requirements

def file_rd(file_name):
    with open(file_name, enconding='utf-8') as rq:
        text = rq.read()
        return text


setup(
    packages=find_packages(exclude=['tests', '*.tests'])
)
