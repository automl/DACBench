import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DAClib",
    version="0.0.1",
    author="Theresa Eimer",
    author_email="eimer@tnt.uni-hannover.de",
    description="Dynamic Algorithm Control benchmark library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/automl/DAClib",
    packages=setuptools.find_packages(),
    classifiers=[
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
    ],
)
