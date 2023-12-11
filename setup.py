#!/usr/bin/env python
from setuptools import setup, find_packages

__version__ = "0.1"

long_description = """Script to build superlattice model for MULTIBINIT """

setup(
    name="MB_SLMAKER",
    version=__version__,
    description="sl_script: script to build superlattice model for MULTIBINIT", 
    long_description=long_description,
    author="Alireza Sasani",
    author_email="alr.sasani@gmail.com",
    license="BSD-2-clause",
    packages=find_packages(),
    scripts=[
    ],
    install_requires=[
        "numpy",
        "scipy",
        "ase>=3.19",
        "spglib",
        "packaging>=20.0",
        "pre-commit",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires=">=3.6",
)
