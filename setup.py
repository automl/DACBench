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

def file_rd(file_name):
    with open(file_name, enconding='utf-8') as rq:
        text = rq.read()
        return text


setup(
    name='DACBench',
    version = '0.0.1',
    author_email = 'eimer@tnt.uni-hannover.de',
    author = 'Theresa Eimer',
    home_page = 'https://www.automl.org/automl/dacbench/',
    project_urls={
        'Documentation' : 'https://dacbench.readthedocs.io/en/latest/',
    'Project Page' : 'https://www.tnt.uni-hannover.de/en/project/dacbench/'
    },
    license = 'Apache 2.0',
    license_file = file_rd('LICENSE'),
    long_description=file_rd('README.md'),

    packages=find_packages(exclude=['tests', 'examples', 'dacbench.wrappers.*', 'dacbench. envs.fast-downward/*']),

    keywords = [
    'DAC',
    'Dynamic Algorithm Configuration',
    'HPO'
    ],

    classifiers = [
     'Programming Language :: Python :: 3',
     'Natural Language :: English',
     'Environment :: Console',
     'Intended Audience :: Developers',
     'Intended Audience :: Education',
     'Intended Audience :: Science/Research',
     'License :: OSI Approved :: Apache Software License',
     'Operating System :: POSIX :: Linux',
     'Topic :: Scientific/Engineering :: Artificial Intelligence',
     'Topic :: Scientific/Engineering',
     'Topic :: Software Development',
    ]

)
