import json
import os
from setuptools import setup, find_packages


def get_extra_requirements():
    """ Helper function to read in all extra requirement files in the extra
        requirement folder. """
    extra_requirements = {}
    for file in os.listdir('./other_requirements'):
        with open(f'./other_requirements/{file}', encoding='utf-8') as fh:
            requirements = json.load(fh)
            extra_requirements.update(requirements)
    return extra_requirements


def read_file(file_name):
    with open(file_name, encoding='utf-8') as fh:
        text = fh.read()
    return text


setup(
    packages=find_packages(exclude=['tests', 'examples'])
)
